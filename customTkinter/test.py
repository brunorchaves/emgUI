import numpy as np
import matplotlib.pyplot as plt

import multiprocessing

from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()

def worker(q):
	m = Myo(mode=emg_mode.PREPROCESSED)
	m.connect()
	
	def add_to_queue(emg, movement):
		q.put(emg)

	m.add_emg_handler(add_to_queue)
	
	def print_battery(bat):
		print("Battery level:", bat)

	m.add_battery_handler(print_battery)

	 # Orange logo and bar LEDs
	m.set_leds([128, 0, 0], [128, 0, 0])
	# Vibrate to know we connected okay
	m.vibrate(1)
	
	"""worker function"""
	while True:
		m.run()
	print("Worker Stopped")

def plot(scr, vals):
	DRAW_LINES = True
    # Generating sample data for 8 channels
    num_channels = 8
    duration = 10  # seconds
    sampling_rate = 1000  # samples per second
    num_samples = duration * sampling_rate

    # Generate random EMG data for each channel
    np.random.seed(42)  # For reproducibility
    emg_data = np.random.randn(num_channels, num_samples)

    # Generate time array
    time = np.arange(num_samples) / sampling_rate

    # Define labels for each channel
    channel_labels = ['Channel A', 'Channel B', 'Channel C', 'Channel D',
                    'Channel E', 'Channel F', 'Channel G', 'Channel H']

    # Define colors for each channel
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

    # Create a figure and subplots
    fig, axs = plt.subplots(num_channels, 1, figsize=(6, 4), sharex=True)

    # Plotting each channel separately
    for i in range(num_channels):
        axs[i].plot(time, emg_data[i], color=colors[i])  # Set color for each channel
        axs[i].set_ylabel("Volts",fontsize = 6)
        axs[i].grid(True)
        # Position the channel label inside the graph
        x_label = time[-1] * 0.95  # x-coordinate for the label
        y_label = np.max(emg_data[i]) * 0.9  # y-coordinate for the label

    # Set common x-axis label and title
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("EMG Signal")

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the plot
    plt.show()

# -------- Main Program Loop -----------
if __name__ == "__main__":
	p = multiprocessing.Process(target=worker, args=(q,))
	p.start()
	# scr = pygame.display.set_mode((w, h))

	try:
		while True:
			# Handle pygame events to keep the window responding
			# pygame.event.pump()
			# Get the emg data and plot it
			while not(q.empty()):
				emg = list(q.get())
				# plot(scr, [e / 500. for e in emg])
				print(emg)

	except KeyboardInterrupt:
		print("Quitting")
		# pygame.quit()
		quit()
