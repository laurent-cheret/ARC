import { useEffect, useState } from 'react';
import styles from './styles.module.scss';
import { getTrainingTask, resetToTrainingTask, stepDemonstration } from './lib/api';
import Grid from './_components/Grid/Grid';
import { StepOutput, Task } from './types/types';
import EnvStep from './_components/EnvStep/EnvStep';

function App() {
  const [taskId, setTaskId] = useState('00d62c1b');
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [stepOutputs, setStepOutputs] = useState<StepOutput[] | null>(null);

  const handleInputChange = (e: any) => {
    setTaskId(e.target.value);
  };

  useEffect(() => {
    const fetchData = async () => {
      const task = await getTrainingTask(taskId);
      setCurrentTask(task);
    };

    fetchData().catch(console.error);
  }, []);

  // Function to handle button click
  const handleClickReset = async () => {
    await resetToTrainingTask(taskId);
    const task = await getTrainingTask(taskId);
    setCurrentTask(task);
  };

  const handleClickStepDemo = async () => {
    const result = await stepDemonstration(taskId);
    console.log(result);

    setStepOutputs(stepOutputs ? [...stepOutputs, result] : [result]);
  };

  return (
    <div className={styles.app}>
      <h1>ARC dataset</h1>

      <div>
        <input type="text" value={taskId} onChange={handleInputChange} placeholder="Task ID" />
        <button onClick={handleClickReset}>Reset to training task</button>
      </div>

      {currentTask && (
        <>
          <h3>Current task : {taskId}</h3>

          <div className="initial-state">
            {currentTask?.train.map((example, index) => (
              <div key={index} className="example-container">
                <div>
                  <h4>input {index}</h4>
                  <Grid key="input" taskGrid={example.input}></Grid>
                </div>
                <div>
                  <h4>output {index}</h4>
                  <Grid key="output" taskGrid={example.output}></Grid>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {stepOutputs &&
        stepOutputs.length > 0 &&
        stepOutputs.map((stepOutput, index) => <EnvStep stepOutput={stepOutput} key={index}></EnvStep>)}

      <button onClick={handleClickStepDemo}>Step demo</button>
    </div>
  );
}

export default App;
