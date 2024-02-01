####################
#### __ IMPORTS ____
####################
# General libraries
import os
import random
import pandas
import numpy as np
from datetime import datetime

# Psychopy. N.B. prefs must be set before importing psychopy modules
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice', 'PTB', 'pyo', 'pygame']
from psychopy import visual, sound, event, core
from psychopy.hardware import keyboard
#defaultKeyboard = keyboard.Keyboard(backend='iohub')
defaultKeyboard = keyboard.Keyboard(backend='psychotoolbox')
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, FINISHED)

##############################
#### __ EXPERIMENT CONFIG ____
##############################

# PARTICIPANT INFO
participant = 'Test'
output_path = os.path.join('./output', participant + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M"))
os.makedirs(output_path, exist_ok=True)
behavioural = pandas.DataFrame(columns={"File", "Block", 'Type_of_question', 
										"Key_given_answer", "Key_correct_answer", "Stim_trial", "Stim_behav",
										'Given_answer', 'Correct_answer', 'Correctness'})

# PARAMETERS DEBUGGING
ask_behavioural = True
play_stimuli = True
full_screen = False 
p_behavioural = 0.2

# PARAMETERS EXP
# presentation
# stim_IDs = [2, 3, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20] # short version (31.2*2+16.5 ~1h20min) 
stim_IDs = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20] # long version (43*2+20 ~1h50min)
folder_input = './input/'
stim_types = ['melody', 'song', 'speech']
nr_blocks = 3
df_time_lengths = pandas.read_csv(os.path.join(folder_input, 'stimuli_lengths.csv'))
df_lyrics_keywords = pandas.read_csv(os.path.join(folder_input, 'lyrics_keywords.csv'))
# audio
volume = 1
sampleRate = 44100 # Hz
startTime_behav = 20  # in sec
stopTime_behav = 28   # in sec
# visual
ratio_text_size = 40
ratio_text_width=2
ratio_cross = 40
contrast = 0.1
# triggers
trigger = False
port_name = 'COM1'

###############################
#### __ GENERATE EXP PARAM ____
###############################
# Genrate checkpoints
nr_stimuli = len(stim_IDs)
blocks_checkpoints = [nr_stimuli*x for x in range(nr_blocks)]

# GENERATE RANDOM STIMULI SEQUENCE FOR THIS SUBJECT AND SAVE IT
# For each stimulus, generate random sequence of [speech, song, melody]
# types_in_blocks: [nr_stimuli x 1] list where each item will be a 
# [1 x 3] list of strings, for instance ["melody", "speech", "song"]
types_in_blocks = []
for i in range(nr_stimuli):
	types_in_block = random.sample(stim_types, nr_blocks)
	# assert that there are no repeted versions of the stimulus (e.g. two melodies)
	assert len(types_in_block) == len(set(types_in_block))
	types_in_blocks.append(types_in_block)

# Generate final stimulus sequence. For each block generate a ordered sequence of IDs.
# Then, for each ID, check the appropriate stimulus type (speech, song or melody) which
# was decided in the previous loop. Shuffle the list and save the final stimulus sequence.
# Save useful metadata: block in which will be played and position in the block sequence.
stim_seq = []
block_seq = []
count_in_block_seq = []
# for each block generate a random list of IDs
for i in range(nr_blocks):
	file_seq_in_block = []
	# create a list of audio file names
	for j, id in enumerate(stim_IDs):
		typ = types_in_blocks[j][i]
		if id > 9:
			name_file = str(id) + '_' + typ + '.wav'
		else:
			name_file = '0' + str(id) + '_' + typ + '.wav'
		file_seq_in_block.append(name_file)
		block_seq.append(i+1)
		count_in_block_seq.append(j+1)
	# randomize the list of file names
	random.shuffle(file_seq_in_block)
	stim_seq = stim_seq + file_seq_in_block
# assert contains unique stimuli file names and no stimulus is repeated twice
assert len(stim_seq) == len(set(stim_seq))

# Save other useful metadata such as stimulus ID, type, time length 
IDs_seq = []
types_seq = []
time_lengths_seq = []
for j, filename in enumerate(stim_seq):
	types_seq.append(filename.split('.')[0][3:])
	IDs_seq.append(filename.split('.')[0][:2])
	time_lengths_seq.append(df_time_lengths.loc[df_time_lengths['File'] == filename]['Length'].values[0])

# Save stimulus sequence and metadata immediatelty
df = pandas.DataFrame(data={"File": stim_seq, "ID": IDs_seq, "Type": types_seq, "Block": block_seq, "Count in block": count_in_block_seq, "Length": time_lengths_seq})
df.to_csv(os.path.join(output_path, 'stimuli_sequence.csv'), sep=',',index=False)

############################
#### __ PSYCHOPY CONFIG ____
############################ 
win = visual.Window(
	size=[800, 600],
	units="pix",
	fullscr=full_screen,
	color=[-1, -1, -1]
)

win.mouseVisible = False

fixation = visual.shape.ShapeStim(win=win, 
								  vertices="cross", 
								  color='white', 
								  fillColor='white', 
								  contrast=contrast, 
								  size=max(win.size)/ratio_cross)

if trigger:
	from psychopy import serial
	# select here the address of your port
	port = serial.Serial(port_name) 

############################
#### __ EXPERIMENT ____
############################ 

# INTRODUCTION
# Slide 1
text = visual.TextStim(win=win, text="Good morning "+ str(participant) + "!\
	\n\nDuring this experiment, you will listen to songs, speech, and melodies.\
	\nThe recorded data will be used to study how our brains process these three types of stimuli that are so special to humans.\
	\n\nPress a key on the keyboard to continue!",\
	color="white",
	contrast=contrast,
	wrapWidth=max(win.size)/ratio_text_width,
	height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False

# Slide 2
text = visual.TextStim(win=win, text="We would like you to pay attention to the audio stimuli as much as possible.\
	\n\nThis means trying to follow the melodies and understand what's said in the lyrics.\
	\nFrom time to time, we will ask you some questions to assess your attention level. Please try to answer the best you can.\
	\n\nPress a key on the keyboard to continue!",\
	color="white",
	contrast=contrast,
	wrapWidth=max(win.size)/ratio_text_width,
	height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False

# Slide 3
text = visual.TextStim(win=win, text="The experiment consists of " + str(nr_blocks) + " blocks, "\
	"each one counting " + str(nr_stimuli) + " stimuli.\
	\n\nBetween blocks, there is a pause of 10-20 minutes when you can stretch and drink.\
	You can also move between trials before pressing the key to starts the new stimulus.\
	\nIf you need to get up, please call the experimenter who will help you as the electrodes are very fragile.\
	\n\n Press a key on the keyboard to continue!",\
	color="white",
	contrast=contrast,
	wrapWidth=max(win.size)/ratio_text_width,
	height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False

# Slide 4
text = visual.TextStim(win=win, text="While the stimuli are played, you should minimize movements and eye blinking as they highly impact the quality of the recordings.\
	\n\nTry to relax and fix the cross that will appear at the center of the screen during the presentation. So it is important that now you find a comfortable position.\
	\n\n Press a key on the keyboard to continue!",\
	color="white",
	contrast=contrast,
	wrapWidth=max(win.size)/ratio_text_width,
	height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False

# Slide 5
text = visual.TextStim(win=win, text="We are almost ready!\
    \n\nPhones can interfere with the recording, so we invite you to leave it outside the cabin or switch it off.\
	\n\nRemember that the experimenter is always outside the cabin, ready to help you in case you need it.\
	\n\nWhen you're ready, press a key to start the experiment!", 
	color="white",
	contrast=contrast,
	wrapWidth=max(win.size)/ratio_text_width,
	height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False

# Stimuli presentation begins
for i, stim in enumerate(stim_seq):

	# Slide intro to block if new block starts
	if i in blocks_checkpoints:	
		text = visual.TextStim(win=win, text="Block nr " + str(block_seq[i]) + " is about to start."\
			"\n\n If you need to move or to take a break, that's the time!"
			"\n\n When you're ready, press a key", 
			color="white",
			contrast=contrast,
			wrapWidth=max(win.size)/ratio_text_width,
			height= max(win.size)/ratio_text_size)
		text.autoDraw = True
		win.flip()
		event.waitKeys()
		text.autoDraw = False

	# Slide intro of each stimulus
	text = visual.TextStim(win=win, text="Block " + str(block_seq[i]) +\
							"\nTrial " + str(count_in_block_seq[i]) + "/" + str(nr_stimuli) +\
							# "\n\nID: " + str(IDs_seq[i]) +\
							# "\nType: " + str(types_seq[i]) +\
							# "\nLength in seconds: " + str(time_lengths_seq[i]) +\
							"\n\n Press a key to listen!",  
							color="white",
							contrast=contrast,
							wrapWidth=max(win.size)/ratio_text_width,
							height= max(win.size)/ratio_text_size)
	text.autoDraw = True
	win.flip()
	event.waitKeys()
	text.autoDraw = False	

	# Draw fixaxtion cross
	fixation.autoDraw = True
	win.flip()
	
	# Play sound and disable keyboard for the time the stimulus is played
	if play_stimuli:
		path_audio = os.path.join(folder_input, "stimuli", stim_seq[i])
		mySound = sound.Sound(path_audio, preBuffer=-1, volume=volume)	
		mySound.play()
		if trigger:
			port.write(str.encode(1))  # chr(IDs_seq[i])
			port.flush()

		# core.wait(time_lengths_seq[i] + 0.1)
		while mySound.status == PLAYING:
			# check for quit (typically the Esc key)
			if defaultKeyboard.getKeys(keyList=["escape"]):
				core.quit()

	fixation.autoDraw = False 

	if ask_behavioural:
		# Ask behavioural questions 20% of the time
		if np.random.choice([1, 0], p=[p_behavioural, 1-p_behavioural]):
			text = visual.TextStim(win=win, text="Attention! Behavioural question now. \
												  Press a key when you're ready!", 
									color="white",
									contrast=contrast,
									wrapWidth=max(win.size)/ratio_text_width,
									height= max(win.size)/ratio_text_size)
			text.autoDraw = True
			win.flip()
			event.waitKeys()
			text.autoDraw = False

			if types_seq[i] == 'speech':	
				type_of_question = 'content'
			elif types_seq[i] == 'melody':	
				type_of_question = 'excerpt'
			elif types_seq[i] == 'song':
				type_of_question = random.choice(['content', 'excerpt'])

			if type_of_question == 'content':
				# Extract first the keywords assosciated with this stimulus, i.e. the right keywords and then other two possible answers
				correct_answer= df_lyrics_keywords.loc[df_lyrics_keywords['ID'] == int(IDs_seq[i])]['Keywords'].values[0]
				possible_answers = list(np.random.choice([x for x in df_lyrics_keywords['Keywords'].values if x not in [correct_answer]], 2))
				
				# Create the answers vector and shuffle it
				possible_answers.append(correct_answer)
				random.shuffle(possible_answers)
				
				# Extract the index of correct answer to compare it to the the answer of the user
				key_correct_answer = str(possible_answers.index(correct_answer) + 1)

				# Ask the question
				text = visual.TextStim(win=win, text="Which group of words best describes the lyrics you just heard?\
									    Type the number of the correct answer using the keyboard" +
									    "\n\n1.) " + possible_answers[0] + 
										"\n2.) " + possible_answers[1] + 
										"\n3.) " + possible_answers[2] + 
										"\n4.) None of the above", 
										color="white",
										contrast=contrast,
										wrapWidth=max(win.size)/ratio_text_width,
										height= max(win.size)/ratio_text_size)
				text.autoDraw = True
				win.flip()

				# Get the response
				key_given_answer = event.waitKeys(keyList=["1", "2", "3", "4"])[0]
				if key_given_answer == '4':
					given_answer = "None of the above"
				else:
					given_answer = possible_answers[int(key_given_answer)-1]
				text.autoDraw = False

				behavioural = behavioural.append({"File":stim_seq[i], 
											"Block":block_seq[i], 
											"Type_of_question":type_of_question, 
											"Key_given_answer":key_given_answer,
											"Key_correct_answer":key_correct_answer,
											"Given_answer":given_answer,
											"Correct_answer":correct_answer,
											"Correctness":(key_given_answer == key_correct_answer)}, ignore_index=True)
			
			else:
				# Decide if to play the correct audio excerpt or the wrong one (coming from another stimulus):
				# First, I randomly sample from all the stimuli except the correct one with probability 1/N.
				# Then I randomly chose between the correct one and te alternative (probability 0.5)
				correct_stim_ID = IDs_seq[i]
				wrong_stim_ID = random.choice([x for x in IDs_seq if x not in [correct_stim_ID]])
				proposed_stim_ID = random.choice([correct_stim_ID, wrong_stim_ID])

				# Save what's the correct answer as a consequence
				if proposed_stim_ID == correct_stim_ID:
					# If the proposed stimulus is the right one, correct answer is YES
					key_correct_answer = '1'  
				else:
					# If the proposed stimulus is NOT the right one, correct answer is N
					key_correct_answer = '2'  

				# Draw fixaxtion cross
				fixation.autoDraw = True
				win.flip()

				# Play the chosen audio excerpt
				path_audio = os.path.join(folder_input, "stimuli", str(proposed_stim_ID) + '_melody.wav')
				mySound = sound.Sound(path_audio, startTime=startTime_behav, stopTime=stopTime_behav)
				mySound.play()
				core.wait((stopTime_behav-startTime_behav) + 0.1)

				# Ask the question
				fixation.autoDraw = False 
				text = visual.TextStim(win=win, text="Was this melody excerpt extracted from the stimulus you just heard?\
								Type the number of the correct answer using the keyboard. \n\n1.) YES" + "\n 2.) NO", 
								color="white",
								contrast=contrast,
								wrapWidth=max(win.size)/ratio_text_width,
								height= max(win.size)/ratio_text_size)
				text.autoDraw = True
				win.flip()

				# Get the response
				key_given_answer = event.waitKeys(keyList=["1", "2"])[0]
				# correct_answer = 'Stim trial ' + str(correct_stim_ID)
				# given_answer   = 'Stim behav ' + str(proposed_stim_ID)
				text.autoDraw = False 					
			
				behavioural = behavioural.append({"File":stim_seq[i], 
												"Block":block_seq[i], 
												"Type_of_question":type_of_question, 
												"Key_given_answer":key_given_answer,
												"Key_correct_answer":key_correct_answer,
												"Stim_trial":str(correct_stim_ID),
												"Stim_behav":str(proposed_stim_ID),
												"Correctness":(key_given_answer == key_correct_answer)}, ignore_index=True)
				
			# Save behavioural responses
			behavioural.to_csv(os.path.join(output_path, 'answers_to_questions.csv'), sep=',',index=False)

### END EXPERIMENT
text = visual.TextStim(win=win, text="Thank you "+ str(participant) + " for your participation!\
					   \n\nIf you'd like, we can keep you updated on the experiment's outcomes.\
					   \n\nPress one final time a key to end the experiment!", 
					   color="white",
					   contrast=contrast,
					   wrapWidth=max(win.size)/ratio_text_width,
					   height= max(win.size)/ratio_text_size)
text.autoDraw = True
win.flip()
event.waitKeys()
text.autoDraw = False
if trigger:
	port.close()
win.close()
