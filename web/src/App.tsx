import { useEffect, useState } from 'react';
import styles from './styles.module.scss';
import {
  getAllTaskIds,
  getClosestTasks,
  getTrainingTask,
  resetToTrainingTask,
  setNewDemoList,
  stepDemonstration,
} from './lib/api';
import Grid from './_components/Grid/Grid';
import { StepOutput, Task } from './types/types';
import EnvStep from './_components/EnvStep/EnvStep';

function App() {
  const [taskId, setTaskId] = useState('00d62c1b');
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [stepOutputs, setStepOutputs] = useState<StepOutput[] | undefined>(undefined);
  const [demoActionList, setDemoActionList] = useState('identify_and_isolate_objects, add, reorder_by_object_size');

  // useEffect(() => {
  //   const fetchData = async () => {
  //     await resetToTrainingTask(taskId);
  //     const task = await getTrainingTask(taskId);
  //     setCurrentTask(task);
  //   };

  //   fetchData().catch(console.error);
  // }, []);

  // useEffect(() => {
  //   const fetchData = async () => {
  //     // await getAllTaskIds();
  //     await getClosestTasks();
  //   };

  //   fetchData();
  // }, []);

  const handleChangedTaskId = (e: any) => {
    setTaskId(e.target.value);
  };
  const handleChangedActionList = (e: any) => {
    setDemoActionList(e.target.value);
  };

  // Function to handle button click
  const handleClickReset = async () => {
    await resetToTrainingTask(taskId);
    const task = await getTrainingTask(taskId);
    setCurrentTask(task);
    setStepOutputs([]);
  };

  const handleClickSetDemoList = async () => {
    const actionsList = demoActionList.split(',').map((action) => action.replace(/"/g, '').trim());
    console.log(actionsList);
    await setNewDemoList(taskId, actionsList);
    setStepOutputs([]);
  };

  const handleClickSingleStep = async () => {
    const result = await stepDemonstration(taskId);
    if (result) {
      console.log(result);
      setStepOutputs(stepOutputs ? [...stepOutputs, result] : [result]);
    }
  };

  const handleClickStepAll = async () => {
    let result: StepOutput | undefined = undefined;
    do {
      result = await stepDemonstration(taskId);
      if (result) {
        setStepOutputs((prevStepOutputs) =>
          prevStepOutputs ? [...prevStepOutputs, result as StepOutput] : [result as StepOutput]
        );
        // window.scrollTo(0, document.body.scrollHeight);
      }
    } while (result);
  };

  return (
    <div className={styles.app}>
      <h1>ARC primitive testing</h1>

      <div>
        <input type="text" value={taskId} onChange={handleChangedTaskId} placeholder="Task ID" />
        <button onClick={handleClickReset}>Reset to original demonstrations.json list</button>
      </div>

      <div className="demo-list">
        <input type="text" value={demoActionList} onChange={handleChangedActionList} placeholder="Action list" />
        <button onClick={handleClickSetDemoList}>Reset to new demonstration list</button>
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

      <button onClick={handleClickSingleStep}>Single Step</button>
      <button onClick={handleClickStepAll}>Step ALL</button>
    </div>
  );
}

export default App;
