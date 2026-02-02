  # Terminal 1 (Learner)
  export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
  export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
  python run_dp_serl_curriculum.py --learner \
      --exp_name=peg_in_hole_square_III \
      --checkpoint_path=/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints_curriculum

  # Terminal 2 (Actor)
  export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
  export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
  python run_dp_serl_curriculum.py --actor \
      --exp_name=peg_in_hole_square_III \
      --ip=localhost \
      --checkpoint_path=/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints_curriculum