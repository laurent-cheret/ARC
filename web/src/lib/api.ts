import { StepOutput, Task } from '../types/types';

// export const API_ROOT = 'http://127.0.0.1:5000';
export const API_ROOT = 'api';

export async function getTrainingTask(taskId: string): Promise<Task> {
  const result = await fetch(`${API_ROOT}/dataset/training/${taskId}`, {
    method: 'GET',
  });
  return await result.json();
}

export async function resetToTrainingTask(taskId: string): Promise<Task> {
  const result = await fetch(`${API_ROOT}/demonstration/reset/${taskId}`, {
    method: 'GET',
  });
  return await result.json();
}

export async function stepDemonstration(taskId: string): Promise<StepOutput> {
  const result = await fetch(`${API_ROOT}/demonstration/step/${taskId}`, {
    method: 'GET',
  });
  const json = await result.json();
  json.current_grids = json.current_grids.map((jsonString: string) => JSON.parse(jsonString));

  return json;
}
