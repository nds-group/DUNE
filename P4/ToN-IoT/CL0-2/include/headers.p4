/* -*- P4_16 -*- */

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

/* Standard ethernet header */
header ethernet_h {
    mac_addr_t   dst_addr;
    mac_addr_t   src_addr;
    ether_type_t ether_type;
}

/*Custom header for notifying the next switch about the classification result */
header notify_h {
    bit<8>       inf_result;
    bit<8>       is_flow_classified;
    bit<8>      pkt_count;
}

/* IPV4 header */
header ipv4_h {
    bit<4>       version;
    bit<4>       ihl;
    bit<8>       diffserv;
    bit<16>      total_len;
    bit<16>      identification;
    bit<3>       flags;
    bit<13>      frag_offset;
    bit<8>       ttl;
    bit<8>       protocol;
    bit<16>      hdr_checksum;
    ipv4_addr_t  src_addr;
    ipv4_addr_t  dst_addr;
}

/* TCP header */
header tcp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<32> seq_no;
    bit<32> ack_no;
    bit<4>  data_offset;
    bit<4>  res;
    bit<1>  cwr;
    bit<1>  ece;
    bit<1>  urg;
    bit<1>  ack;
    bit<1>  psh;
    bit<1>  rst;
    bit<1>  syn;
    bit<1>  fin;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

/* UDP header */
header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> udp_total_len;
    bit<16> checksum;
}

// header tna_timestamps_h {
    // bit<16> pad_1;
    // bit<48> ingress_mac;
    // bit<16> pad_2;
    // bit<48> ingress_global;
    // bit<14> pad_3;
    // bit<18> enqueue;
    // bit<14> pad_4;
    // bit<18> dequeue_delta;
    // bit<16> pad_5;
    // bit<48> egress_global;
    // bit<16> pad_6;
    // bit<48> egress_tx;
// }

/***********************  H E A D E R S  ************************/
struct my_ingress_headers_t {
    ethernet_h   ethernet;
    // recirc_h     recirc;
    ipv4_h       ipv4;
    tcp_h        tcp;
    udp_h        udp;
    notify_h notify;

}


/******  G L O B A L   I N G R E S S   M E T A D A T A  *********/
struct my_ingress_metadata_t {
    bit<1> is_first;
    bit<8> classified_flag;

    bit<32> flow_ID;
    bit<(INDEX_WIDTH)> register_index;

    bit<16> hdr_srcport;
    bit<16> hdr_dstport;
    bit<4> tcp_hdr_len;
    bit<16> tcp_windows_size;
    bit<16> udp_len;
    bit<16> pkt_len_max;
    bit<16> pkt_len_min;
    bit<16> pkt_len_total;
    bit<8> ip_proto;
    // bit<1> tcp_flag_ack;

    bit<8> pkt_count;
    bit<32> time_last_pkt;
    bit<16> total_len;
    // bit<32> flow_duration;

    bit<8> class0;
    bit<8> class1;
    bit<8> class2;
    bit<8> class3;
    bit<8> class_model1;
    bit<8> class_model2;
    bit<8> final_class;

    bit<84> codeword0;
    bit<40> codeword1;
    bit<40> codeword2;
    bit<40> codeword3;

    bit<8> digest_info; // used for either class or collision info

    bit<8> f_action; // For flow_action table
    bit<2> is_store;
    bit<2> is_refresh;
    bit<2> is_flow;

    // tna_timestamps_h tna_timestamps_hdr;
    // ptp_metadata_t tx_ptp_md_hdr;

    // bit<48> measured_latency;
}

struct flow_class_digest {  // maximum size allowed is 47 bytes
    
    ipv4_addr_t  source_addr;   // 32 bits
    ipv4_addr_t  destin_addr;   // 32 bits
    bit<16> source_port;
    bit<16> destin_port;
    bit<8> protocol;
    bit<8> class_value;
    bit<8> packet_num;
    bit<(INDEX_WIDTH)> register_index; // To send register index info to the controller
    bit<2> is_refresh;
    bit<2> is_store;
    bit<2> is_flow;
    // total size is 2*2 + 2*4 + 2 + (16/8==2) = 16 bytes           
}

// struct timestamp_digest {  // maximum size allowed is 47 bytes
    
//     bit<48>  latency;
//     // total size is = 6 bytes           
// }

