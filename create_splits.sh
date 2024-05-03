midi_dir=data/midi
token_dir=data/token
for split in train valid test
	do python tools/generate_token_data_by_file_list.py data/meta/${split}.txt $token_dir $split_dir ;
done
