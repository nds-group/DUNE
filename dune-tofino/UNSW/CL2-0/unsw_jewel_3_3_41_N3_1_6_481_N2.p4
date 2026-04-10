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
        meta.tcp_windows_size = hdr.tcp.window;
        meta.udp_len = 0;
        // transition accept;
        transition parse_notify;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.hdr_dstport = hdr.udp.dst_port;
        meta.hdr_srcport = hdr.udp.src_port;
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
    // First Model
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_classified_flag_model1;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model1)
    read_classified_flag_model1 = {
        void apply(inout bit<8> classified_flag, out bit<8> output) {
            output = classified_flag;
        }
    };
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model1)
    update_classified_flag_model1 = {
        void apply(inout bit<8> classified_flag) {
            classified_flag = meta.final_class;
        }
    };
    // Second model
    Register<bit<8>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_classified_flag_model2;
    /* Register read action */
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model2)
    read_classified_flag_model2 = {
        void apply(inout bit<8> classified_flag, out bit<8> output) {
            output = classified_flag;
        }
    };
    RegisterAction<bit<8>,bit<(INDEX_WIDTH)>,bit<8>>(reg_classified_flag_model2)
    update_classified_flag_model2 = {
        void apply(inout bit<8> classified_flag) {
            classified_flag = meta.class_model2;
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

    Register<bit<32>,bit<(INDEX_WIDTH)>>(MAX_REGISTER_ENTRIES) reg_flow_iat_min;
    /* Register read action */
    RegisterAction<bit<32>,bit<(INDEX_WIDTH)>,bit<32>>(reg_flow_iat_min)
    read_flow_iat_min = {
        void apply(inout bit<32> flow_iat_min, out bit<32> output) {
            if (meta.pkt_count <= 2){
                flow_iat_min = meta.iat;
            }
            else if(meta.iat < flow_iat_min){
                flow_iat_min = meta.iat;
            }
            output = flow_iat_min;
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
        meta.class0 = classe;
    }
    action SetClass1(bit<8> classe) {
        meta.class1 = classe;
    }
    action SetClass2(bit<8> classe) {
        meta.class2 = classe;
    }
    action SetClass3(bit<8> classe) {
        meta.class_model2 = classe;
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
        meta.class_model1 = class_result;
    }

    action set_default_class() {
    }
    
    action set_flow_feats() {
        meta.flow_iat_min = 0;
        meta.flow_iat_max = 0;
    }

    /* Feature table actions */
    // First model - [[9, 11, 20], [13, 7, 20], [15, 7, 18]]
    action SetCode0(bit<9> code0, bit<13> code1, bit<15> code2) {
        meta.codeword0[39:31] = code0;
        meta.codeword1[39:27] = code1;
        meta.codeword2[39:25] = code2;
    }
    action SetCode1(bit<11> code0, bit<7> code1, bit<7> code2) {
        meta.codeword0[30:20] = code0;
        meta.codeword1[26:20] = code1;
        meta.codeword2[24:18] = code2;
    }
    action SetCode2(bit<20> code0, bit<20> code1, bit<18> code2) {
        meta.codeword0[19:0] = code0;
        meta.codeword1[19:0] = code1;
        meta.codeword2[17:0] = code2;
    }
    
    // Second model - [[67, 69, 105, 90, 63, 86]]
    /* Feature table actions */
    action SetCode3(bit<67> code0) {
        meta.codeword3[479:413] = code0;
    }
    action SetCode4(bit<69> code0) {
        meta.codeword3[412:344] = code0;
    }
    action SetCode5(bit<105> code0) {
        meta.codeword3[343:239] = code0;
    }
     action SetCode6(bit<90> code0) {
        meta.codeword3[238:149] = code0;
    }
    action SetCode7(bit<63> code0) {
        meta.codeword3[148:86] = code0;
    }
    action SetCode8(bit<86> code0) {
        meta.codeword3[85:0] = code0;
    }
    

    action set_flow_action(bit<8> f_action) {
        meta.f_action = f_action;
    }
    action set_def_flow_action() {
        meta.f_action = 34;
        drop();
    }
    
    // First Model FEATURES: ['udp.length' 'Flow IAT Max' 'Flow IAT Min']
    /* Feature tables */ 
    table table_feature0{
        key = {meta.udp_len: range @name("feature0");}
	    actions = {@defaultonly nop; SetCode0;}
	    size = 25;
        const default_action = nop();
	}
    table table_feature1{
	    key = {meta.flow_iat_max[31:29]: range @name("feature1");}
	    actions = {@defaultonly nop; SetCode1;}
	    size = 10;
        const default_action = nop();
	}
    table table_feature2{
        // [4,0]
	    key = {meta.flow_iat_min[31:21]: range @name("feature2");}
	    actions = {@defaultonly nop; SetCode2;}
	    size = 35;
        const default_action = nop();
	}

    // Second Model FEATURES: ['udp.length' 'ip.len' 'dstport' 'srcport' 'ip.ttl' 'tcp.window_size_value']
    /* Feature tables */
    table table_feature3{
        key = {meta.udp_len: range @name("feature3");}
	    actions = {@defaultonly nop; SetCode3;}
	    size = 70;
        const default_action = nop();
	}
    table table_feature4{
	    key = {meta.total_len: range @name("feature4");}
	    actions = {@defaultonly nop; SetCode4;}
	    size = 70;
        const default_action = nop();
	}
    table table_feature5{
	    key = {meta.hdr_dstport: range @name("feature5");}
	    actions = {@defaultonly nop; SetCode5;}
	    size = 100;
        const default_action = nop();
	}
    table table_feature6{
	    key = {meta.hdr_srcport: range @name("feature6");}
	    actions = {@defaultonly nop; SetCode6;}
	    size = 80;
        const default_action = nop();
	}
    table table_feature7{
	    key = {hdr.ipv4.ttl: range @name("feature7");}
	    actions = {@defaultonly nop; SetCode7;}
	    size = 50;
        const default_action = nop();
	}
    table table_feature8{
	    key = {meta.tcp_windows_size: range @name("feature8");}
	    actions = {@defaultonly nop; SetCode8;}
	    size = 90;
        const default_action = nop();
	}


    /* Code tables */
	table code_table0{
	    key = {meta.codeword0: ternary;}
	    actions = {@defaultonly nop; SetClass0;}
	    size = 41;
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
	    size = 481;
        const default_action = nop();
	}

    table voting_table {
        key = {
            meta.class0: exact;
            meta.class1: exact;
            meta.class2: exact;
        }
        actions = {set_final_class; @defaultonly set_default_class;}
        size = 32;
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
                    meta.flow_iat_min = read_flow_iat_min.execute(meta.register_index);
                } //END OF CHECK ON IF NO COLLISION
            } // END OF CHECK ON WHETHER FIRST CLASS
            meta.is_flow = 0;
            if (meta.pkt_count < 4){
                if(meta.pkt_count < 3){   // Set flow level features as 0 if the packet before N1th, where N1 = the inference point of the first model
                    set_flow_feats();
                }
                // ** First Model **
                // apply feature tables to assign codes
                table_feature0.apply();
                table_feature1.apply();
                table_feature2.apply();

                // apply code tables to assign labels
                code_table0.apply();
                code_table1.apply();
                code_table2.apply(); 

                voting_table.apply(); // It sets class_model1: the class from the first model
                
                // ** Second Model **
                table_feature3.apply();
                table_feature4.apply();
                table_feature5.apply();
                table_feature6.apply();
                table_feature7.apply();
                table_feature8.apply();

                code_table3.apply();

                meta.class_model2 = meta.class_model2 + 1;   // It sets class_model2: the class from the second model

                meta.is_refresh = 0;
                hdr.notify.is_flow_classified = 0;
                if (meta.pkt_count < 3){    // If the packet count is less than N1+1
                    update_classified_flag_model2.execute(meta.register_index); // store the result of second model
                    if (meta.class_model1 == 2){ // OTHERS: If the packet classified as Others in the first model, set the final class with the result coming from the second model
                        meta.final_class =  meta.class_model2;
                    }
                    else{  // ONE OF THE CLASSES: : If the packet classified as one of the classes in the first model, set the final class with the result coming from the first model
                        meta.final_class =  meta.class_model1;
                    }
                }
                else{
                    meta.classified_flag = read_classified_flag_model2.execute(meta.register_index);    // since the N2 (2) < N1 (3), we already stored the result and here we read it
                    if (meta.class_model1 == 2){ // OTHERS
                        meta.final_class =  meta.classified_flag;
                    }
                    else{  // ONE OF THE CLASSES
                        meta.final_class =  meta.class_model1;
                    }
                    // update_classified_flag_model1.execute(meta.register_index);  // Commented because of the stage issue
                }
                if (meta.pkt_count == 3){
                    meta.is_refresh = 1; // Store the result and refresh the memory
                    meta.is_flow = 1;
                    if (meta.final_class != 19){
                        hdr.notify.is_flow_classified = 1; // Notify the downstream switches about tft the flow is classified
                    }
                }
                // ** SET CLASS and NOTIFICATION DATA, and ACTIVATE DIGEST **
                hdr.notify.inf_result = meta.final_class;
                // hdr.notify.pkt_count = meta.pkt_count;
                meta.is_store = 1;
                ig_dprsr_md.digest_type = 1;        // activating the digest after classification
                ipv4_forward(24);
            }
            // else {   // Commented because of the stage issue
            //     meta.f_action = read_classified_flag_model1.execute(meta.register_index);
            // } 
            // }
        }
        if (meta.f_action == 19) {  // If the flow is classified as OTHERS (obtained by the flow_action table)
            hdr.notify.inf_result = meta.f_action;
            hdr.notify.is_flow_classified = 0;
            // hdr.notify.pkt_count = 4;
            ipv4_forward(24);
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
