/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

#include "./include/types.p4"
#include "./include/headers.p4"
/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/
parser TofinoIngressParser(
        packet_in pkt,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            1 : parse_resubmit;
            0 : parse_port_metadata;
        }
    }
    state parse_resubmit {
        // Parse resubmitted packet here.
        transition reject;
    }
    state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition accept;
    }
}

parser IngressParser(packet_in        pkt,
    /* User */
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            TYPE_IPV4:  parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        meta.total_len = hdr.ipv4.total_len;
        // meta.ip_proto  = (bit<16>) hdr.ipv4.protocol;
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP:  parse_tcp;
            TYPE_UDP:  parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        meta.hdr_dstport = hdr.tcp.dst_port;
        meta.hdr_srcport = hdr.tcp.src_port;
        meta.tcp_hdr_len = hdr.tcp.data_offset;
        meta.tcp_windows_size = hdr.tcp.window;
        meta.udp_len = 0;
        meta.tcp_flag_ack = hdr.tcp.ack;
        meta.tcp_flag_psh = hdr.tcp.psh;
        transition parse_notify;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.hdr_dstport = hdr.udp.dst_port;
        meta.hdr_srcport = hdr.udp.src_port;
        meta.tcp_hdr_len = 0;
        meta.tcp_windows_size = 0;
        meta.udp_len = hdr.udp.udp_total_len;
        meta.tcp_flag_ack = 0;
        meta.tcp_flag_psh = 0;
        transition parse_notify;
    }

    state parse_notify {
       pkt.extract(hdr.notify);
       transition accept;
    }
}

/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
/***************** M A T C H - A C T I O N  *********************/
control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    /* Registers for flow management */
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_classified_flag;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag)
    // update_classified_flag = {
    read_classified_flag = {
        void apply(inout bit<8> classified_flag, out bit<8> output) {
            output = classified_flag;
        }
    };
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag)
    // update_classified_flag = {
    update_classified_flag = {
        void apply(inout bit<8> classified_flag) {
            classified_flag = hdr.notify.inf_result;
        }
    };

    Register<bit<32>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_flow_ID;
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_flow_ID)
    update_flow_ID = {
        void apply(inout bit<32> flow_ID) {
            flow_ID = meta.flow_ID;
        }
    };
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_flow_ID)
    read_only_flow_ID = {
        void apply(inout bit<32> flow_ID, out bit<32> output) {
            output = flow_ID;
        }
    };

    Register<bit<32>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_time_last_pkt;
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_time_last_pkt)
    read_time_last_pkt = {
        void apply(inout bit<32> time_last_pkt, out bit<32> output) {
            output = time_last_pkt;
            time_last_pkt = ig_prsr_md.global_tstamp[31:0];
        }
    };

    //registers for ML inference - features
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_pkt_count;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_pkt_count)
    read_pkt_count = {
        void apply(inout bit<8> pkt_count, out bit<8> output) {
            pkt_count = pkt_count + 1;
            output = pkt_count;
        }
    };

    Register<bit<16>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_pkt_len_max;
    /* Register read action */
    RegisterAction<bit<16>,bit<(INDEX_WIDTH)>,bit<16>>(reg_pkt_len_max)
    read_pkt_len_max = {
        void apply(inout bit<16> pkt_len_max, out bit<16> output) {
            if (meta.is_first == 1){
                pkt_len_max = hdr.ipv4.total_len;
            }
            else if (hdr.ipv4.total_len > pkt_len_max){
                pkt_len_max  = hdr.ipv4.total_len;
            }
            output = pkt_len_max;
        }
    };

    Register<bit<16>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_pkt_len_min;
    /* Register read action */
    RegisterAction<bit<16>,bit<(INDEX_WIDTH)>,bit<16>>(reg_pkt_len_min)
    read_pkt_len_min = {
        void apply(inout bit<16> pkt_len_min, out bit<16> output) {
            if (meta.is_first == 1){
                pkt_len_min = hdr.ipv4.total_len;
            }
            else if (hdr.ipv4.total_len < pkt_len_min){
                pkt_len_min  = hdr.ipv4.total_len;
            }
            output = pkt_len_min;
        }
    };

    Register<bit<16>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_pkt_len_total;
    /* Register read action */
    RegisterAction<bit<16>,bit<(INDEX_WIDTH)>,bit<16>>(reg_pkt_len_total)
    read_pkt_len_total = {
        void apply(inout bit<16> pkt_len_total, out bit<16> output) {
            if (meta.is_first == 1){
                pkt_len_total = hdr.ipv4.total_len;
            }
            else{
                pkt_len_total = pkt_len_total + hdr.ipv4.total_len;
            }
            output = pkt_len_total;
        }
    };
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_ack_flag_count;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_ack_flag_count)
    read_ack_flag_count = {
        void apply(inout bit<8> ack_flag_count, out bit<8> output) {
            if (meta.tcp_flag_ack == 1){
                ack_flag_count = ack_flag_count + 1;
            }
            output = ack_flag_count;
        }
    };

    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_psh_flag_count;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_psh_flag_count)
    read_psh_flag_count = {
        void apply(inout bit<8> psh_flag_count, out bit<8> output) {
            if (meta.tcp_flag_psh == 1){
                psh_flag_count = psh_flag_count + 1;
            }
            output = psh_flag_count;
        }
    };

    /* Declaration of the hashes*/
    Hash<bit<32>>(HashAlgorithm_t.CRC32)              flow_id_calc;
    Hash<bit<(INDEX_WIDTH)>>(HashAlgorithm_t.CRC16)   idx_calc;

    /* Calculate hash of the 5-tuple to represent the flow ID */
    action get_flow_ID(bit<16> srcPort, bit<16> dstPort) {
        meta.flow_ID = flow_id_calc.get({hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr,srcPort, dstPort, hdr.ipv4.protocol});
    }
    /* Calculate hash of the 5-tuple to use as 1st register index */
    action get_register_index(bit<16> srcPort, bit<16> dstPort) {
        meta.register_index = idx_calc.get({hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr,srcPort, dstPort, hdr.ipv4.protocol});
    }

    /* Assign class if at leaf node */
    action SetClass0(bit<8> classe) {
        hdr.notify.inf_result = classe;
    }

    /* Forward to a specific port upon classification */
    action ipv4_forward(PortId_t port) {
        meta.f_action = 0;
        ig_tm_md.ucast_egress_port = port;
    }
    /* Custom Do Nothing Action */
    action nop(){}

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
        meta.f_action = 0;
    }

    action set_final_class(bit<8> class_result) {
        meta.final_class = class_result;
    }
    
    action set_flow_feats() {
        meta.pkt_len_max = 0;
        meta.pkt_len_min = 0;
        meta.pkt_len_total = 0;
        meta.ack_flag_count = 0;
        meta.psh_flag_count = 0;
        meta.is_flow = 0;
    }

    // [[25, 9, 17, 11, 3, 2, 17, 4, 12, 6, 9, 13]]
    /* Feature table actions */
    action SetCode0(bit<25> code0) {
        meta.codeword0[127:103] = code0;
    }
    action SetCode1(bit<9> code0) {
        meta.codeword0[102:94] = code0;
    }
    action SetCode2(bit<17> code0) {
        meta.codeword0[93:77] = code0;
    }
    action SetCode3(bit<11> code0) {
        meta.codeword0[76:66] = code0;
    }
    action SetCode4(bit<3> code0) {
        meta.codeword0[65:63] = code0;
    }
    action SetCode5(bit<2> code0) {
        meta.codeword0[62:61] = code0;
    }
    action SetCode6(bit<17> code0) {
        meta.codeword0[60:44] = code0;
    }
    action SetCode7(bit<4> code0) {
        meta.codeword0[43:40] = code0;
    }
    action SetCode8(bit<12> code0) {
        meta.codeword0[39:28] = code0;
    }
    action SetCode9(bit<6> code0) {
        meta.codeword0[27:22] = code0;
    }
    action SetCode10(bit<9> code0) {
        meta.codeword0[21:13] = code0;
    }
    action SetCode11(bit<13> code0) {
        meta.codeword0[12:0] = code0;
    }

    action set_flow_action(bit<8> f_action) {
        meta.f_action = f_action;
    }
    action set_def_flow_action() {
        meta.f_action = 34;
        drop();
    }
    
    // FEATURES: ['ip.len' 'udp.length' 'Packet Length Total' 'dstport' 'tcp.flags.ack'
                // 'PSH Flag Count' 'Max Packet Length' 'ACK Flag Count' 'Min Packet Length'
                // 'tcp.hdr_len' 'ip.ttl' 'tcp.window_size_value']
    /* Feature tables */
    table table_feature0{
	    key = {meta.total_len: range @name("feature0");}
	    actions = {@defaultonly nop; SetCode0;}
	    size = 25;
        const default_action = nop();
	}
    table table_feature1{
	    key = {meta.udp_len: range @name("feature1");}
	    actions = {@defaultonly nop; SetCode1;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature2{
	    key = {meta.pkt_len_total: range @name("feature2");}
	    actions = {@defaultonly nop; SetCode2;}
	    size = 20;
        const default_action = nop();
	}
    table table_feature3{
        key = {meta.hdr_dstport: range @name("feature3");}
	    actions = {@defaultonly nop; SetCode3;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature4{
        key = {meta.tcp_flag_ack: range @name("feature4");}
	    actions = {@defaultonly nop; SetCode4;}
	    size = 5;
        const default_action = nop();
	}
    table table_feature5{
	    key = {meta.psh_flag_count: range @name("feature5");}
	    actions = {@defaultonly nop; SetCode5;}
	    size = 5;
        const default_action = nop();
	}
    table table_feature6{
	    key = {meta.pkt_len_max: range @name("feature6");}
	    actions = {@defaultonly nop; SetCode6;}
	    size = 20;
        const default_action = nop();
	}
    table table_feature7{
	    key = {meta.ack_flag_count: range @name("feature7");}
	    actions = {@defaultonly nop; SetCode7;}
	    size = 5;
        const default_action = nop();
	}
    table table_feature8{
	    key = {meta.pkt_len_min: range @name("feature8");}
	    actions = {@defaultonly nop; SetCode8;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature9{
	    key = {meta.tcp_hdr_len: range @name("feature9");}
	    actions = {@defaultonly nop; SetCode9;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature10{
	    key = {hdr.ipv4.ttl: range @name("feature10");}
	    actions = {@defaultonly nop; SetCode10;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature11{
	    key = {meta.tcp_windows_size: range @name("feature11");}
	    actions = {@defaultonly nop; SetCode11;}
	    size = 15;
        const default_action = nop();
	}


    /* Code tables */
	table code_table0{
	    key = {meta.codeword0: ternary;}
	    actions = {@defaultonly nop; SetClass0;}
	    size = 129;
        const default_action = nop();
	}

    /* Forwarding-Inference Block Table */
    table flow_action_table {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            meta.hdr_srcport: exact;
            meta.hdr_dstport: exact;
            meta.ip_proto: exact;
        }
        actions = {set_flow_action; @defaultonly set_def_flow_action;}
        // size = 25000;
        size = 60500;
        const default_action = set_def_flow_action();
    }


    apply {
	    meta.ip_proto  = (bit<16>) hdr.ipv4.protocol;
        flow_action_table.apply();
        bit<32> tmp_flow_ID;
        //compute flow_ID and hash index
        get_flow_ID(meta.hdr_srcport, meta.hdr_dstport);
        get_register_index(meta.hdr_srcport, meta.hdr_dstport);

        // Forward, if flow is already classified. Otherwise, run model first.
        // Code here to execute if table experienced a hit
        if (meta.f_action == 50) {
            if (hdr.notify.is_flow_classified == 1){
                meta.pkt_count = 3;  // to make it clear the register
                meta.is_store = 0;
                meta.final_class = 1;
                tmp_flow_ID = read_only_flow_ID.execute(meta.register_index);
                if(meta.flow_ID == tmp_flow_ID){        // no hash collision
                    ig_dprsr_md.digest_type = 1;        // activating the digest after classification
                }
            }
            else {
                // modify timestamp register
                meta.time_last_pkt = read_time_last_pkt.execute(meta.register_index);
                // check if register array is empty
                if (meta.time_last_pkt == 0){ // we do not yet know this flow
                    meta.is_first = 1;
                    update_flow_ID.execute(meta.register_index);
                    meta.pkt_count = read_pkt_count.execute(meta.register_index);
                    meta.pkt_len_max = read_pkt_len_max.execute(meta.register_index);
                    meta.pkt_len_min = read_pkt_len_min.execute(meta.register_index);
                    meta.pkt_len_total = read_pkt_len_total.execute(meta.register_index);
                    meta.ack_flag_count = read_ack_flag_count.execute(meta.register_index);
                    meta.psh_flag_count = read_psh_flag_count.execute(meta.register_index);

                }
                else { // not the first packet - get flow_ID from register
                    meta.is_first = 0;
                    tmp_flow_ID = read_only_flow_ID.execute(meta.register_index);
                    if(meta.flow_ID != tmp_flow_ID){ // hash collision
                        meta.pkt_count = 0;
                    }
                    else { // not first packet and not hash collision
                        meta.pkt_count = read_pkt_count.execute(meta.register_index);
                        meta.pkt_len_max = read_pkt_len_max.execute(meta.register_index);
                        meta.pkt_len_min = read_pkt_len_min.execute(meta.register_index);
                        meta.pkt_len_total = read_pkt_len_total.execute(meta.register_index);
                        meta.ack_flag_count = read_ack_flag_count.execute(meta.register_index);
                        meta.psh_flag_count = read_psh_flag_count.execute(meta.register_index);
                    } //END OF CHECK ON IF NO COLLISION
                } // END OF CHECK ON WHETHER FIRST CLASS
                if (hdr.notify.inf_result == 2){
                    // hdr.notify.is_flow_classified = 0;
                    meta.is_store = 1;
                    meta.is_flow = 1;
                    if (meta.pkt_count < 4){
                        if(meta.pkt_count < 3){
                            set_flow_feats();
                        }
                        // apply feature tables to assign codes
                        table_feature0.apply();
                        table_feature1.apply();
                        table_feature2.apply();
                        table_feature3.apply();
                        table_feature4.apply();
                        table_feature5.apply();
                        table_feature6.apply();
                        table_feature7.apply();
                        table_feature8.apply();
                        table_feature9.apply();
                        table_feature10.apply();
                        table_feature11.apply();

                        // apply code tables to assign labels
                        code_table0.apply();

                        hdr.notify.inf_result = hdr.notify.inf_result + 1;
                        update_classified_flag.execute(meta.register_index);

                        /* Commented to handle stage issues */
                        // hdr.notify.pkt_count = meta.pkt_count;
                        // 
                        // To refresh the register after classification 
                        // if (hdr.notify.inf_result == 5) {
                        //     hdr.notify.is_flow_classified = 0;
                        // }

                        ig_dprsr_md.digest_type = 1;        // activating the digest after classification
                    }
                    else {
                        /* Adapted to handle stage issues: Here, we make the packet always forwarded to the next switch untill the forwarding table is updated. */
                        hdr.notify.inf_result = 5;
                        // meta.f_action  = read_classified_flag.execute(meta.register_index); // commented to handle the stage issue
                    } 
                }
            }
            ipv4_forward(160);
        }  
        if (meta.f_action == 5) {
            hdr.notify.inf_result = 5;
            hdr.notify.is_flow_classified = 0;
            hdr.notify.pkt_count = 4;
            ipv4_forward(160);
        }
    } //END OF APPLY
} //END OF INGRESS CONTROL

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    // Checksum() ipv4_checksum;

    Digest<flow_class_digest>() digest;

    apply {

        if (ig_dprsr_md.digest_type == 1) {
            digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, meta.hdr_srcport, meta.hdr_dstport,meta.ip_proto, hdr.notify.inf_result, meta.pkt_count, meta.register_index, meta.is_flow, meta.is_store});
        }

        /* we do not update checksum because we used ttl field for stats*/
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
        pkt.emit(hdr.notify);
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/
#include "./include/egress.p4"

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;
