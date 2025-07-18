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
        // meta.tcp_flag_ack = hdr.tcp.ack;
        meta.udp_len = 0;
        // transition accept;
        transition parse_notify;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.hdr_dstport = hdr.udp.dst_port;
        meta.hdr_srcport = hdr.udp.src_port;
        // meta.tcp_flag_ack = 0;
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
    // First model
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_classified_flag_model1;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model1)
    // update_classified_flag = {
    read_classified_flag_model1 = {
        void apply(inout bit<8> classified_flag, out bit<8> output) {
            output = classified_flag;
        }
    };
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model1)
    // update_classified_flag = {
    update_classified_flag_model1 = {
        void apply(inout bit<8> classified_flag) {
            // if (meta.pkt_count == 3){
            classified_flag = meta.class_model1;
            // }

        }
    };
    // Second model
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_classified_flag_model2;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model2)
    // update_classified_flag = {
    read_classified_flag_model2 = {
        void apply(inout bit<8> classified_flag, out bit<8> output) {
            output = classified_flag;
        }
    };
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model2)
    // update_classified_flag = {
    update_classified_flag_model2 = {
        void apply(inout bit<8> classified_flag) {
            // if (meta.pkt_count == 3){
            classified_flag = meta.final_class;
            // }

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
    action SetClass2(bit<8> classe) {
        meta.class2 = classe;
    }
    action SetClass3(bit<8> classe) {
        meta.class3 = classe;
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
        // meta.final_class = class_result;
        meta.class_model2 = class_result;
    }

    action set_default_class() {
    }
    
    action set_flow_feats() {
        meta.pkt_len_max = 0;
        meta.pkt_len_min = 0;
        meta.pkt_len_total = 0;
    }

    /* Feature table actions */
    // First model - [21, 5, 11, 13, 14, 20]]
    action SetCode0(bit<21> code0) {
        meta.codeword0[83:63] = code0;
    }
    action SetCode1(bit<5> code0) {
        meta.codeword0[62:58] = code0;
    }
    action SetCode2(bit<11> code0) {
        meta.codeword0[57:47] = code0;
    }
     action SetCode3(bit<13> code0) {
        meta.codeword0[46:34] = code0;
    }
    action SetCode4(bit<14> code0) {
        meta.codeword0[33:20] = code0;
    }
    action SetCode5(bit<20> code0) {
        meta.codeword0[19:0] = code0;
    }
    // Second model 
    // [[10, 5, 8, 4, 13], [10, 2, 14, 8, 6], [10, 6, 10, 9, 5]]
    /* Feature table actions */
    action SetCode6(bit<10> code0, bit<10> code1, bit<10> code2) {
        meta.codeword1[39:30] = code0;
        meta.codeword2[39:30] = code1;
        meta.codeword3[39:30] = code2;
    }
    action SetCode7(bit<5> code0, bit<2> code1, bit<6> code2) {
        meta.codeword1[29:25] = code0;
        meta.codeword2[29:28] = code1;
        meta.codeword3[29:24] = code2;
    }
    action SetCode8(bit<8> code0, bit<14> code1, bit<10> code2) {
        meta.codeword1[24:17] = code0;
        meta.codeword2[27:14] = code1;
        meta.codeword3[23:14] = code2;
    }
    action SetCode9(bit<4> code0, bit<8> code1, bit<9> code2) {
        meta.codeword1[16:13] = code0;
        meta.codeword2[13:6] = code1;
        meta.codeword3[13:5] = code2;
    }
    action SetCode10(bit<13> code0, bit<6> code1, bit<5> code2) {
        meta.codeword1[12:0] = code0;
        meta.codeword2[5:0] = code1;
        meta.codeword3[4:0] = code2;
    }


    action set_flow_action(bit<8> f_action) {
        meta.f_action = f_action;
    }
    action set_def_flow_action() {
        meta.f_action = 34;
        drop();
    }
    
    // FEATURES: ['dstport' 'ip.ttl' 'Min Packet Length' 'tcp.window_size_value' 'tcp.hdr_len' 'srcport']
    /* Feature tables */
    table table_feature0{
        key = {meta.hdr_dstport: range @name("feature0");}
	    actions = {@defaultonly nop; SetCode0;}
	    size = 25;
        const default_action = nop();
	}
    table table_feature1{
	    key = {hdr.ipv4.ttl: range @name("feature1");}
	    actions = {@defaultonly nop; SetCode1;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature2{
	    key = {meta.pkt_len_min: range @name("feature2");}
	    actions = {@defaultonly nop; SetCode2;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature3{
	    key = {meta.tcp_windows_size: range @name("feature3");}
	    actions = {@defaultonly nop; SetCode3;}
	    size = 20;
        const default_action = nop();
	}
    table table_feature4{
	    key = {meta.tcp_hdr_len: range @name("feature4");}
	    actions = {@defaultonly nop; SetCode4;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature5{
        key = {meta.hdr_srcport: range @name("feature5");}
	    actions = {@defaultonly nop; SetCode5;}
	    size = 30;
        const default_action = nop();
	}
    // 
    // FEATURES: ['dstport' 'tcp.hdr_len' 'tcp.window_size_value' 'Max Packet Length' 'Packet Length Total']
    /* Feature tables */
    table table_feature6{
        key = {meta.hdr_dstport: range @name("feature6");}
	    actions = {@defaultonly nop; SetCode6;}
	    size = 15;
        const default_action = nop();
	}
    table table_feature7{
	    key = {meta.tcp_hdr_len: range @name("feature7");}
	    actions = {@defaultonly nop; SetCode7;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature8{
	    key = {meta.tcp_windows_size: range @name("feature8");}
	    actions = {@defaultonly nop; SetCode8;}
	    size = 30;
        const default_action = nop();
	}
    table table_feature9{
	    key = {meta.pkt_len_max: range @name("feature9");}
	    actions = {@defaultonly nop; SetCode9;}
	    size = 30;
        const default_action = nop();
	}
    table table_feature10{
	    key = {meta.pkt_len_total: range @name("feature10");}
	    actions = {@defaultonly nop; SetCode10;}
	    size = 30;
        const default_action = nop();
	}


    /* Code tables */
	table code_table0{
	    key = {meta.codeword0: ternary;}
	    actions = {@defaultonly nop; SetClass0;}
	    size = 85;
        const default_action = nop();
	}
	table code_table1{
        key = {meta.codeword1: ternary;}
	    actions = {@defaultonly nop; SetClass1;}
	    size = 41;
        const default_action = nop();
	}
	table code_table2{
        key = {meta.codeword2: ternary;}
	    actions = {@defaultonly nop; SetClass2;}
	    size = 41;
        const default_action = nop();
	}
    table code_table3{
        key = {meta.codeword3: ternary;}
	    actions = {@defaultonly nop; SetClass3;}
	    size = 41;
        const default_action = nop();
	}

    table voting_table {
        key = {
            meta.class1: exact;
            meta.class2: exact;
            meta.class3: exact;
        }
        actions = {set_final_class; @defaultonly set_default_class;}
        size = 64;
        const default_action = set_default_class();
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
        // size = 25000;
        size = 63000;
        const default_action = set_def_flow_action();
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
                if(meta.flow_ID == tmp_flow_ID){ // No hash collision
                    meta.is_refresh = 1;  // to make it clear the register but do not store the result
                }
                else{
                    meta.is_refresh = 0;  // to make it clear the register but do not store the result
                }
                meta.is_store = 0;
                meta.pkt_count = hdr.notify.pkt_count;
                meta.final_class = hdr.notify.inf_result;
                // do not store the result but clear the register
                ig_dprsr_md.digest_type = 1;        // activating the digest after classification
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
                        meta.pkt_len_max = read_pkt_len_max.execute(meta.register_index);
                        meta.pkt_len_min = read_pkt_len_min.execute(meta.register_index);
                        meta.pkt_len_total = read_pkt_len_total.execute(meta.register_index);
                    } //END OF CHECK ON IF NO COLLISION
                } // END OF CHECK ON WHETHER FIRST CLASS
                meta.is_flow = 0;
                if (meta.pkt_count < 5){
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

                    // apply code tables to assign labels
                    code_table0.apply();
                    meta.class0 = meta.class0 + 4;
                    meta.class_model1 = meta.class_model1 + 4;

                    // ** Second Model **
                    if(meta.pkt_count < 4){
                        set_flow_feats();
                    }
                    table_feature6.apply();
                    table_feature7.apply();
                    table_feature8.apply();
                    table_feature9.apply();
                    table_feature10.apply();

                    code_table1.apply();
                    code_table2.apply();
                    code_table3.apply();

                    voting_table.apply();
                    meta.class_model2 = meta.class_model2 + 5;
                
                    meta.is_refresh = 0;
                    // If the packet count is less than N1+1 where N1 = the inference point of the first model
                    if (meta.pkt_count < 4){
                        update_classified_flag_model1.execute(meta.register_index);
                        // If the packet classified as Others in the first model
                        if (meta.class0 == 6){ // OTHERS
                            set_final_class(meta.class1);
                        }
                        else{  // ONE OF THE CLASSES
                            set_final_class(meta.class0);
                            if (meta.pkt_count == 3){
                                meta.is_refresh = 1; // Store the result and refresh the memory
                                meta.is_flow = 1;
                            }
                        }
                    }
                    else{
                        meta.class_model1 = read_classified_flag_model1.execute(meta.register_index);
                        // If the packet classified as Others in the first model
                        if (meta.class_model1 == 6){ // OTHERS
                            set_final_class(meta.class1);
                        }
                        else{  // ONE OF THE CLASSES
                            set_final_class(meta.class_model1);
                        }
                        if (meta.pkt_count == 4){
                            meta.is_refresh = 1; // Store the result and refresh the memory
                            meta.is_flow = 1;
                        }
                        update_classified_flag_model2.execute(meta.register_index);
                    } 

                    // SET CLASS and NOTIFICATION DATA
                    if (hdr.notify.inf_result < 5){ // If the flow is classified as one of the classes in upstream switches.
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
                // else {
                //     meta.f_action = read_classified_flag_model2.execute(meta.register_index);
                // } 
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
            
            // digest.pack({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, meta.hdr_srcport, meta.hdr_dstport, hdr.ipv4.protocol, meta.final_class, hdr.notify.inf_result, meta.pkt_count, meta.register_index});
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
