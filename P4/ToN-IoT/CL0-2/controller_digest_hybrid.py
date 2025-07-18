#!/usr/bin/python3

from __future__ import print_function

import os
import sys
import pdb

SDE_INSTALL   = os.environ['SDE_INSTALL']
SDE_PYTHON2   = os.path.join(SDE_INSTALL, 'lib', 'python2.7', 'site-packages')
sys.path.append(SDE_PYTHON2)
sys.path.append(os.path.join(SDE_PYTHON2, 'tofino'))

PYTHON3_VER   = '{}.{}'.format(
                    sys.version_info.major,
                    sys.version_info.minor)
SDE_PYTHON3   = os.path.join(SDE_INSTALL, 'lib', 'python' + PYTHON3_VER, 'site-packages')
sys.path.append(SDE_PYTHON3)
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino'))
sys.path.append(os.path.join(SDE_PYTHON3, 'tofino', 'bfrt_grpc'))

import grpc
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as bfrt_client

import time
import socket, struct

# filename_out = 'ToN-IoT_CL2_CLID0.csv'
filename_out = sys.argv[1]

#
# Connect to the BF Runtime Server
#
interface = bfrt_client.ClientInterface(
    grpc_addr = 'localhost:50052',
    client_id = 1,
    device_id = 0)
print('Connected to BF Runtime Server')

#
# Get the information about the running program
#
bfrt_info = interface.bfrt_info_get()
print('The target runs the program ', bfrt_info.p4_name_get())
#
# Establish that you are using this program on the given connection
#
interface.bind_pipeline_config(bfrt_info.p4_name_get())

# learn_filter_lat = bfrt_info.learn_get("lat_digest")
learn_filter = bfrt_info.learn_get("digest")
# learn_filter2 = bfrt_info.learn_get("digest3")

# List of registers
registers = ['Ingress.reg_flow_ID','Ingress.reg_time_last_pkt','Ingress.reg_pkt_count', 'Ingress.reg_classified_flag_model1', 'Ingress.reg_classified_flag_model2', 'Ingress.reg_pkt_len_max', 'Ingress.reg_pkt_len_min', 'Ingress.reg_pkt_len_total']
# Register to test
# reg_name = registers[0]

#
# Getting info about one specific table
#
flow_act_tbl = bfrt_info.table_get('Ingress.flow_action_table')
print('Flow-action table:', flow_act_tbl)

# getting info about one specific register
# reg_tbl = bfrt_info.table_get(reg_name)
# print('Register max packet length info:', reg_tbl)

# Target pipe_id=0xffff
target = bfrt_client.Target(device_id=0, pipe_id=0xffff)

# header = 'source_addr,destin_addr,source_port,destin_port,protocol,ground_truth,flow_class'
header = 'source_addr,destin_addr,source_port,destin_port,protocol,pkt_count,is_flow,flow_packet_class'

# latency_list = []
count = 0

with open(filename_out, "w") as text_file:
    text_file.write(header)
    text_file.write("\n")

flow_counter = 0
# collision_counter = 0
# total_time = 0
# total_digests = 0
while True:

    # digest = interface.digest_get(timeout=120)
    try:
        digest = interface.digest_get(timeout=2000)
    except:
        f = open("x.txt", "a")
        f.write('---- \n')
        f.close()
        break

    # time_dgst1 = time.time()

    recv_target = digest.target

    # print("RECEIVED", recv_target)
    

    # print("\n\nDigest received from:")
    # print("device id: ", recv_target.device_id)
    # print("pipe id: ", recv_target.pipe_id)

    # digest_type = 0
    # try:
    #     data_list_lat = learn_filter_lat.make_data_list(digest)
    #     digest_type = 2
    # except:
    digest_type = 1
    data_list = learn_filter.make_data_list(digest)
    
    # print('Digest type: ', digest_type)

    
    # print("\n\nDigest received with length: ", len(data_list_lat))
    if digest_type == 1:
        count = count + 1

        # print('Latency values so far: ', latency_list)
        flow_counter = flow_counter + len(data_list)
        # print("\nFlows counted so far: ", flow_counter)

        keys_reg = {'Ingress.reg_flow_ID': [], 'Ingress.reg_time_last_pkt': [],
                    'Ingress.reg_pkt_count': [], 'Ingress.reg_classified_flag_model1': [], 
                    'Ingress.reg_classified_flag_model2': [], 'Ingress.reg_pkt_len_min': [],
                    'Ingress.reg_pkt_len_max': [], 'Ingress.reg_pkt_len_total': []}
        datas_reg = {'Ingress.reg_flow_ID': [], 'Ingress.reg_time_last_pkt': [],
                    'Ingress.reg_pkt_count': [], 'Ingress.reg_classified_flag_model1': [], 
                    'Ingress.reg_classified_flag_model2': [], 'Ingress.reg_pkt_len_min': [],
                    'Ingress.reg_pkt_len_max': [], 'Ingress.reg_pkt_len_total': []}
        keys_table = []
        datas_table = []
        # total_digests = total_digests + len(data_list)
        for dd in data_list:
            data_dict = dd.to_dict()
            # print("\nDigest info: ", data_dict)
            # print(dd)
            # convert ip address into normal format
            source_addr = socket.inet_ntoa(struct.pack('!L', data_dict['source_addr']))
            destin_addr = socket.inet_ntoa(struct.pack('!L', data_dict['destin_addr']))
            source_port = str(data_dict['source_port'])
            destin_port = str(data_dict['destin_port'])
            protocol = str(data_dict['protocol'])
            flow_packet_class = data_dict['class_value']
            pkt_count = str(data_dict['packet_num'])
            register_index = data_dict['register_index']
            
            
            # print(flow_ packet_class, type(flow_packet_class))
            
            # if flow_packet_class == 34:
            #     print()
                # print(source_addr + ',' + destin_addr + ',' + source_port + ',' + destin_port + ',' + protocol + ',' + pkt_count + ',' + str(flow_packet_class))
            
            # if flow_packet_class == 3:
            #     csv_row = source_addr + ',' + destin_addr + ',' + source_port + ',' + destin_port + ',' + protocol + ',' + pkt_count + ',' + str(32)
            # else:
            if (data_dict['is_store'] == 1):
                csv_row = source_addr + ',' + destin_addr + ',' + source_port + ',' + destin_port + ',' + protocol + ',' + pkt_count + ',' + str(flow_packet_class)

            else:
                csv_row = source_addr + ',' + destin_addr + ',' + source_port + ',' + destin_port + ',' + protocol + ',' + pkt_count + ',' + str(55)
            
            with open(filename_out, "a") as text_file:
                        text_file.write(csv_row)
                        text_file.write("\n")
            csv_row_1 = source_addr + ',' + destin_addr + ',' + source_port + ',' + destin_port + ',' + protocol
            if (csv_row_1 == '192.168.1.36,192.168.1.152,45154,80,6'):
                print('is_store: ', data_dict['is_store'], 'pkt_cnt: ', pkt_count, 'result: ', str(flow_packet_class))

            # if flow_class == '255':
            #     print('######### collision #########')
            #     collision_counter = collision_counter + 1
            #     # print(collision_counter)

            # else:
                # keys_reg.append(reg_tbl.make_key([bfrt_client.KeyTuple('$REGISTER_INDEX', register_index)]))
            # if (data_dict['packet_num'] == 4):
            if (data_dict['is_refresh'] == 1):
                if (csv_row_1 == '192.168.1.36,192.168.1.152,45154,80,6'):
                    print('REFRESH is_store: ', data_dict['is_store'], 'pkt_cnt: ', pkt_count, 'result: ', str(flow_packet_class))
                keys_table.append(flow_act_tbl.make_key(
                                [bfrt_client.KeyTuple('hdr.ipv4.src_addr', data_dict['source_addr']), bfrt_client.KeyTuple('hdr.ipv4.dst_addr', data_dict['destin_addr']), 
                                bfrt_client.KeyTuple('meta.hdr_dstport', data_dict['destin_port']), bfrt_client.KeyTuple('meta.hdr_srcport', data_dict['source_port']),
                                bfrt_client.KeyTuple('hdr.ipv4.protocol', data_dict['protocol'])]))
                
                # If the Flow is classified as Others
                # f_action == 3 : Classified flow as one of the classes
                # f_action == 2 : Classified flow as Others
                # if flow_packet_class == 32:
                #     # print(flow_packet_class, type(flow_packet_class))
                #     datas_table.append(flow_act_tbl.make_data([
                #                             bfrt_client.DataTuple('f_action', 2)
                #                         ], 'Ingress.set_flow_action'))
                # else:
                #     print(flow_packet_class, type(flow_packet_class))
                datas_table.append(flow_act_tbl.make_data([
                                    bfrt_client.DataTuple('f_action', flow_packet_class)
                                ], 'Ingress.set_flow_action'))

                
                # if data_dict['class_value'] == 6:
                #     datas_table.append(flow_act_tbl.make_data([
                #                             bfrt_client.DataTuple('port', 260)
                #                         ], 'Ingress.ipv4_forward'))
                    
                # # if data_dict['class_value'] < 6:
                # #     datas_table.append(flow_act_tbl.make_data([
                # #                             bfrt_client.DataTuple('port', 260)
                # #                         ], 'Ingress.drop_cl'))
                # if data_dict['class_value'] < 6:
                #     datas_table.append(flow_act_tbl.make_data([
                #                         ], 'Ingress.drop'))

                for reg_name in registers:
                    reg_tbl = bfrt_info.table_get(reg_name)
                    keys_reg[reg_name].append(reg_tbl.make_key([bfrt_client.KeyTuple('$REGISTER_INDEX', register_index)]))
                    datas_reg[reg_name].append(reg_tbl.make_data([bfrt_client.DataTuple(reg_name+'.f1', 0)]))


        # try:
        # print(csv_row)
        flow_act_tbl.entry_mod(target, keys_table, datas_table, p4_name=bfrt_info.p4_name_get())
        # except: 
        #     print('ERROR: modify')
        for reg_name in registers:
            # print(reg_name)
            reg_tbl = bfrt_info.table_get(reg_name)
            reg_tbl.entry_mod(target, key_list=keys_reg[reg_name], data_list=datas_reg[reg_name], flags={"from_hw":True}, p4_name=bfrt_info.p4_name_get())

