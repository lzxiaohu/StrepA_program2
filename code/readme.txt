Readme.md

Section 1: obtain a simulation bank
1. run the file "simulations_bank.py" to get a simulation bank of 500,000 simulations 
 The corresponding command: "nohup python -u simulations_bank.py> ./log/simulations_bank.log 2>&1 &"
 The outputs: "simulation_banks/simulation_bank_part_*.h5"
 It will cost about 28 hours when 62 vCPU
 
2. Check if all simulations are reasonable and remove the simulations with all zero cases. And then calculate their summary statistics
 run the command: "nohup python -u calculate_summary_stats.py> ./log/calculate_summary_stats.log 2>&1 &"
 The outputs: a. "all_summary_statistics_clean.h5"
						  b. "valid_indices.csv"
 It will cost about 2 minutes when 62 vCPU
 
3. recheck the summary statistics are reasonable and valid
 run the command: "nohup python -u check_nan_values.py> ./log/check_nan_values.log 2>&1 &"
 
 
section 2: calibrate to real observation 
4. Obtain the summary statistics for the real observation 
	run the command "nohup python -u generate_standard_point_fixed.py> ./log/generate_standard_point_fixed.log 2>&1 &"
	The outputs: "standard_points.csv" 
	costs: 20 minutes when 62 vCPU
	
5. Normalize the summary statistics of the real observation and get optimal weights
 run the command "nohup python -u best_weights_for_observations.py> ./log/best_weights_for_observations.log 2>&1 &"
 The outputs: a. "optimal_weights_results_observations.csv"
							 b. "summary_stats.csv"
							c. "summary_stats_normalized.csv"
							d. "R0.csv"
							e. "sigma.csv"
							
6. get scatter plots when trying different epsilon based on the optimal weights
		run the command: "nohup python -u scatter_plots_observations.py> ./log/scatter_plots_observations.log 2>&1 &"
		The outputs: a: "dists_observations_recal.csv"
								b: "***.png"
		Notice: need to modify the value of vmin and vmax
		
7. get the posterior values of R0 and sigma for the real observation,and posterior concentration
	   run the command: "nohup python -u calculate_posterior_concentration_observations.py> ./log/calculate_posterior_concentration_observations.log 2>&1 &"
			the outputs: a. "posterior_scatter_contour_*.png"
										b. "posterior concentration values"
										
8. get the scatter figures and heat maps of selected simulations and compare them to the real observation by visual perception
		run the command: "nohup python -u display_selected_samples.py> ./log/display_selected_samples.log 2>&1 &"
		the outputs: a. "selected_samples_*.csv"
								b. "matrices_heatmap_grid.png"
								c. "matrices_scatter_grid.png"
		Notice: we need to run the command 2 times: first time to get best 20 simulations; second to selected best 10 cases
		
		
9. select the best case most similar to the real observation and get their heat map: a single simulation vs the real observation
   run the command: "nohup python -u plot_single_vs_observation.py> ./log/plot_single_vs_observation.log 2>&1 &" 
	the outputs: "comparison_scatter_id*.png"
	
10. generate a synthetical data based on posterior values of R0 and sigma, and plot its heat map vs the real observation
   run the command: "nohup python -u plot_single_synthetic_vs_observation.py> ./log/plot_single_synthetic_vs_observation.log 2>&1 &"
		the outputs: "comparison_scatter_id0.png"	   


section 3: Identifiability analysis
For instance R0=1.2 sigma=0.94

11. Obtain the summary statistics for synthetical data:R0:{};
sigma:{1.2
	run the command "nohup python -u generate_standard_point_fixed.py> ./log/generate_standard_point_fixed.log 2>&1 &"
	The outputs: "standard_points.csv" 
	costs: 20 minutes when 62 vCPU
	

12. Normalize the summary statistics of a single synthetical case and get optimal weights
 run the command "nohup python -u best_weights_for_sigma0p94_R01p2.py> ./log/best_weights_for_sigma0p94_R01p2.log 2>&1 &"
 The outputs: a. "optimal_weights_results_sigma0p94_R01p2.csv"
							 b. "summary_stats.csv"
							c. "summary_stats_normalized.csv"
							d. "R0.csv"
							e. "sigma.csv"
	
13. get scatter plots when trying different epsilon based on the optimal weights
		run the command: "nohup python -u scatter_plots_for_sigma0p94_R01p2.py> ./log/scatter_plots_for_sigma0p94_R01p2.log 2>&1 &"
		The outputs: a: "dists_sigma0p94_R01p2_recal.csv"
								b: "***.png"
		Notice: need to modify the value of vmin and vmax
		
		
14. get the posterior values of R0 and sigma for synthe,and posterior concentration
	   run the command: "nohup python -u calculate_posterior_concentration_sigma0p94_R01p2.py> ./log/calculate_posterior_concentration_sigma0p94_R01p2.log 2>&1 &"
			the outputs: a. "posterior_scatter_contour_*.png"
										b. "posterior concentration values"
										
15. get the distribution of peak estimates about 20 different random seeds.
		 run the command: "nohup python -u flow_process_sigma0p94_R01p2.py> ./log/flow_process_sigma0p94_R01p2.log 2>&1 &"
		 the outputs: 
									a. "peak_estimates_sigma0p94_R01p2.csv"
									b "flow_process_sigma0p94_R01p2.png"

repeat step 11-15 for all synthetical cases.
After finishing all synthetical data 

16. get visual results for identifiability analysis
  run the command: "nohup python -u plot_identifiability_spread.py> ./log/plot_identifiability_spread.log 2>&1 &"
	the outputs: "ellipse_sigma*_R0*.png"
	
17. get dentifiability analysis data 
run the command: "nohup python -u aggregate_identifiability_results.py> ./log/aggregate_identifiability_results.log 2>&1 &"
 the outputs: a. "identifiability_summary.csv"
							b. "table_s1_identifiability.tex"
							c. "identifiability_summary_compact.csv"
							d. "table_identifiability_compact.tex"

