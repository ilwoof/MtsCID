# SMD
python main.py --run_times 5 --anomaly_ratio 0.5 --dataset SMD --data_path ./data/SMD/ --input_c 38 --d_model 38 --output_c 38

# SMAP
python main.py --run_times 5 --anomaly_ratio 1.0 --dataset SMAP --data_path ./data/SMAP/ --input_c 25 --d_model 25 --output_c 25

# PSM
python main.py --run_times 5 --anomaly_ratio 1.0 --dataset PSM --data_path ./data/PSM/ --input_c 25 --d_model 25 --output_c 25

# MSL
python main.py --run_times 5 --anomaly_ratio 1.0 --dataset MSL --data_path ./data/MSL/ --input_c 55 --d_model 55 --output_c 55

# SWaT
python main.py --run_times 5 --anomaly_ratio 0.1 --dataset SWaT --data_path ./data/SWaT/ --input_c 51 --d_model 51 --output_c 51

# NIPS_TS_Swan
python main.py --run_times 5 --anomaly_ratio 0.9 --dataset NIPS_TS_Swan  --data_path ./data/NIPS_TS_Swan/  --input_c 38 --d_model 38 --output_c 38

# NIPS_TS_Water
python main.py --run_times 5 --anomaly_ratio 1.0 --dataset NIPS_TS_Water --data_path ./data/NIPS_TS_GECCO/ --input_c 9 --d_model 9 --output_c 9
