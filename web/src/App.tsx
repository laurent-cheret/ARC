import { useEffect, useState } from 'react';
import styles from './styles.module.scss';
import {
  getPrimitiveNames,
  resetToTrainingTask,
  setNewDemoList,
  stepAllDemonstration,
  stepDemonstration,
} from './lib/api';
import { StepOutput, Task } from './types/types';
import { useStickyState } from './hooks/useStickyState';
import InputTaskSetter from './_components/InputTaskSetter/InputTaskSetter';
import TaskExamples from './_components/TaskExamples/TaskExamples';
import AutoCompleteChipList from './_components/AutoComplete/AutoComplete';
import EnvStepper from './_components/EnvStepper/EnvStepper';

function App() {
  const [taskId, setTaskId] = useStickyState('00d62c1b', 'currentTaskId');
  const [currentTask, setCurrentTask] = useState<Task | undefined>(undefined);
  const [stepOutputs, setStepOutputs] = useState<StepOutput[]>([]);

  const [primitives, setPrimitives] = useState<string[]>([]);
  const [demoActionList, setDemoActionList] = useState<string[]>([]);

  const [loadingDemo, setLoadingDemo] = useState(false);

  // taskId update
  useEffect(() => {
    const fetchData = async () => {
      const newTask = await resetToTrainingTask(taskId);
      setCurrentTask(newTask);
      setDemoActionList(newTask.demoActions);
      setStepOutputs([]);
    };

    fetchData().catch(console.error);
  }, [taskId]);

  // fetch primitives list
  useEffect(() => {
    const fetchData = async () => {
      const res = await getPrimitiveNames();
      setPrimitives(res.primitives);
    };

    fetchData().catch(console.error);
  }, []);

  const handleStepAll2 = async () => {
    setLoadingDemo(true);
    await setNewDemoList(taskId, demoActionList);

    const result = await stepAllDemonstration(taskId);
    // console.log(result);

    setLoadingDemo(false);
    if (result) {
      setStepOutputs(result);
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
      }
    } while (result);
  };

  return (
    <div className={styles.app}>
      <div className="main">
        <h1 className="title-main display-medium" style={{ marginTop: 0 }}>
          🕹️ ARCade Warriors DSL playground
        </h1>

        <InputTaskSetter taskId={taskId} setTaskId={setTaskId}></InputTaskSetter>

        <AutoCompleteChipList
          primitives={primitives}
          demoActions={demoActionList}
          setDemoActions={setDemoActionList}
        ></AutoCompleteChipList>

        <TaskExamples task={currentTask}></TaskExamples>

        <button className="btn-primary" style={{ marginBottom: '30px' }} onClick={handleStepAll2}>
          Set actions and run demo
        </button>

        <EnvStepper loadingDemo={loadingDemo} stepOutputs={stepOutputs}></EnvStepper>
      </div>
    </div>
  );
}

export default App;
