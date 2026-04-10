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
        meta.ip_proto  = hdr.ipv4.protocol;
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
        // transition accept;
        transition parse_notify;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.hdr_dstport = hdr.udp.dst_port;
        meta.hdr_srcport = hdr.udp.src_port;
        meta.tcp_hdr_len = 0;
        meta.tcp_windows_size = 0;
        meta.udp_len = hdr.udp.udp_total_len;
        // transition accept;
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
            classified_flag = meta.final_class;
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

    Register<bit<32>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_flow_iat_max;
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_flow_iat_max)
    read_flow_iat_max = {
        void apply(inout bit<32> flow_iat_max, out bit<32> output) {
            if (meta.is_first != 1){
                if(meta.iat > flow_iat_max){
                    flow_iat_max = meta.iat;
                }
            }
            output = flow_iat_max;
        }
    };

    Register<bit<32>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_flow_duration;
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_flow_duration)
    read_flow_duration = {
        void apply(inout bit<32> flow_duration, out bit<32> output) {
            if (meta.is_first != 1){
                flow_duration = flow_duration + meta.iat;
            }
            output = flow_duration;
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
        // meta.class0 = classe;
        meta.class0 = classe;
        meta.class_model1 = classe;
    }
    action SetClass1(bit<8> classe) {
        meta.class1 = classe;
    }

    /* Forward to a specific port upon classification */
    action ipv4_forward(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
    }
    /* Custom Do Nothing Action */
    action nop(){}

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    action set_final_class(bit<8> class_result) {
        meta.final_class = class_result;
    }
    
    action set_flow_feats() {
        meta.flow_duration = 0;
        meta.flow_iat_max = 0;
    }

    // [[26, 81, 23, 23, 14, 34, 15]] - [[32, 12, 10, 32, 42]]
    /* Feature table actions */
    action SetCode0(bit<26> code0) {
        meta.codeword0[215:190] = code0;
    }
    action SetCode1(bit<81> code0) {
        meta.codeword0[189:109] = code0;
    }
    action SetCode2(bit<23> code0) {
        meta.codeword0[108:86] = code0;
    }
    action SetCode3(bit<23> code0) {
        meta.codeword0[85:63] = code0;
    }
    action SetCode4(bit<14> code0) {
        meta.codeword0[62:49] = code0;
    }
    action SetCode5(bit<34> code0) {
        meta.codeword0[48:15] = code0;
    }
    action SetCode6(bit<15> code0) {
        meta.codeword0[14:0] = code0;
    }
    // 
    action SetCode7(bit<32> code0) {
        meta.codeword1[127:96] = code0;
    }
    action SetCode8(bit<12> code0) {
        meta.codeword1[95:84] = code0;
    }
    action SetCode9(bit<10> code0) {
        meta.codeword1[83:74] = code0;
    }
    action SetCode10(bit<32> code0) {
        meta.codeword1[73:42] = code0;
    }
    action SetCode11(bit<42> code0) {
        meta.codeword1[41:0] = code0;
    }


    action set_flow_action(bit<8> f_action) {
        meta.f_action = f_action;
    }
    action set_def_flow_action() {
        meta.f_action = 34;
        drop();
    }
    
    // FEATURES: ['ip.len' 'dstport' 'udp.length' 'Flow Duration' 'Flow IAT Max' 'srcport' 'tcp.window_size_value']
    /* Feature tables */
    table table_feature0{
	    key = {meta.total_len: range @name("feature0");}
	    actions = {@defaultonly nop; SetCode0;}
	    size = 35;
        const default_action = nop();
	}
    table table_feature1{
        key = {meta.hdr_dstport: range @name("feature1");}
	    actions = {@defaultonly nop; SetCode1;}
	    size = 85;
        const default_action = nop();
	}
    table table_feature2{
	    key = {meta.udp_len: range @name("feature2");}
	    actions = {@defaultonly nop; SetCode2;}
	    size = 25;
        const default_action = nop();
	}
    table table_feature3{
        key = {meta.flow_duration[31:19]: range @name("feature3");}
	    actions = {@defaultonly nop; SetCode3;}
	    size = 30;
        const default_action = nop();
	}
    table table_feature4{
        key = {meta.flow_iat_max[31:24]: range @name("feature4");}
	    actions = {@defaultonly nop; SetCode4;}
	    size = 20;
        const default_action = nop();
	}
    table table_feature5{
	    key = {meta.hdr_srcport: range @name("feature5");}
	    actions = {@defaultonly nop; SetCode5;}
	    size = 40;
        const default_action = nop();
	}
    table table_feature6{
        key = {meta.tcp_windows_size: range @name("feature6");}
	    actions = {@defaultonly nop; SetCode6;}
	    size = 15;
        const default_action = nop();
	}
    //  ['ip.len' 'ip.ttl' 'tcp.window_size_value' 'srcport' 'dstport']
    table table_feature7{
	    key = {meta.total_len: range @name("feature7");}
	    actions = {@defaultonly nop; SetCode7;}
	    size = 30;
        const default_action = nop();
	}
    table table_feature8{
	    key = {hdr.ipv4.ttl: range @name("feature8");}
	    actions = {@defaultonly nop; SetCode8;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature9{
        key = {meta.tcp_windows_size: range @name("feature9");}
	    actions = {@defaultonly nop; SetCode9;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature10{
	    key = {meta.hdr_srcport: range @name("feature10");}
	    actions = {@defaultonly nop; SetCode10;}
	    size = 40;
        const default_action = nop();
	}
    table table_feature11{
	    key = {meta.hdr_dstport: range @name("feature11");}
	    actions = {@defaultonly nop; SetCode11;}
	    size = 50;
        const default_action = nop();
	}


    /* Code tables */
	table code_table0{
	    key = {meta.codeword0: ternary;}
	    actions = {@defaultonly nop; SetClass0;}
	    size = 217;
        const default_action = nop();
	}
	table code_table1{
        key = {meta.codeword1: ternary;}
	    actions = {@defaultonly nop; SetClass1;}
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
            hdr.ipv4.protocol: exact;
        }
        actions = {set_flow_action; @defaultonly set_def_flow_action;}
        size = 63000;
        const default_action = set_def_flow_action();
    }
    /* Compute packet interarrival time (IAT)*/
    action get_iat_value(){
        meta.iat = ig_prsr_md.global_tstamp[31:0] - meta.time_last_pkt;
    }

    apply {
        flow_action_table.apply();
        // Forward, if flow is already classified as Others. Otherwise, run model.
        bit<32> tmp_flow_ID;
        //compute flow_ID and hash index
        get_flow_ID(meta.hdr_srcport, meta.hdr_dstport);
        get_register_index(meta.hdr_srcport, meta.hdr_dstport);
        // code here to execute if table experienced a hit
        if (meta.f_action == 50) {

            if (hdr.notify.is_flow_classified == 1){
                tmp_flow_ID = read_only_flow_ID.execute(meta.register_index);
                meta.is_refresh = 0;
                meta.is_store = 0;
                meta.pkt_count = hdr.notify.pkt_count;
                meta.final_class = hdr.notify.inf_result;
                // do not store the result but clear the register
                ig_dprsr_md.digest_type = 1;        // activating the digest after classification
            }
            else {
                
                // modify timestamp register
                meta.time_last_pkt = read_time_last_pkt.execute(meta.register_index);
                // calculate iat
                get_iat_value();
                
                // check if register array is empty
                if (meta.time_last_pkt == 0){ // we do not yet know this flow
                    meta.is_first = 1;
                    update_flow_ID.execute(meta.register_index);
                    meta.pkt_count = read_pkt_count.execute(meta.register_index);
                
                }
                else { // not the first packet - get flow_ID from register
                    meta.is_first = 0;
                    tmp_flow_ID = read_only_flow_ID.execute(meta.register_index);
                    if(meta.flow_ID != tmp_flow_ID){ // hash collision
                        meta.pkt_count = 0;
                    }
                    else { // not first packet and not hash collision
                        //read and update packet count
                        meta.pkt_count = read_pkt_count.execute(meta.register_index);
                        meta.flow_iat_max = read_flow_iat_max.execute(meta.register_index);
                        meta.flow_duration = read_flow_duration.execute(meta.register_index);

                    } //END OF CHECK ON IF NO COLLISION
                } // END OF CHECK ON WHETHER FIRST CLASS
                meta.is_flow = 0;
                if (meta.pkt_count < 4){
                    if(meta.pkt_count < 3){
                        set_flow_feats();
                    }
                    // ** First Model **
                    // apply feature tables to assign codes
                    table_feature0.apply();
                    table_feature1.apply();
                    table_feature2.apply();
                    table_feature3.apply();
                    table_feature4.apply();
                    table_feature5.apply();
                    table_feature6.apply();

                    // apply code tables to assign labels
                    code_table0.apply();
                    meta.class0 = meta.class0 + 21;
                    meta.class_model1 = meta.class_model1 + 21;

                    // ** Second Model **
                    table_feature7.apply();
                    table_feature8.apply();
                    table_feature9.apply();
                    table_feature10.apply();
                    table_feature11.apply();

                    code_table1.apply();
                    meta.class1 = meta.class1 + 23;

                    meta.is_refresh = 0;
                    if (meta.class0 == 24){ // OTHERS
                        set_final_class(meta.class1);
                    }
                    else{  // ONE OF THE CLASSES
                        set_final_class(meta.class0);
                    }
                    if (meta.pkt_count == 3){
                        update_classified_flag.execute(meta.register_index);
                        meta.is_flow = 1;
                        meta.is_refresh = 1; // Store the result and refresh the memory
                        hdr.notify.is_flow_classified = 1; // Notify the upstream switches about tft the flow is classified
                    }

                    // SET CLASS and NOTIFICATION DATA
                    if (hdr.notify.inf_result < 22){
                        // If the flow is classified as Others, just refresh the memory if necessary but do not store the result
                        meta.is_store = 0; // do not store the result
                    }
                    else {
                        // If the flow is classified as Others in the downstream switch, tag with the result obtained in the current switch.
                        hdr.notify.inf_result = meta.final_class;
                        hdr.notify.pkt_count = meta.pkt_count;
                        meta.is_store = 1;
                    }
                    ig_dprsr_md.digest_type = 1;        // activating the digest after classification
                }
                else {
                    meta.f_action = read_classified_flag.execute(meta.register_index);
                } 
            }
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
            digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, meta.hdr_srcport, meta.hdr_dstport, hdr.ipv4.protocol, meta.final_class, meta.pkt_count, meta.register_index, meta.is_refresh, meta.is_store, meta.is_flow});
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
