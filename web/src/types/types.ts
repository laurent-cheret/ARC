export type TaskGrid = number[][];

export type Example = {
  input: TaskGrid;
  output: TaskGrid;
};

export type Task = {
  train: Example[];
  test: Example[];
  demoActions: string[];
};

export type StepOutput = {
  step: number;
  action_name: string;
  current_grids: TaskGrid[][];
  memory_grids: TaskGrid[][];
};
