import { useState } from 'react';
import styles from './styles.module.scss';

export default function InputTaskSetter({ taskId, setTaskId }: { taskId: string; setTaskId: Function }) {
  const [inputValue, setInputValue] = useState(taskId);

  const handleSubmit = () => {
    setTaskId(inputValue);
  };

  return (
    <div className={styles.inputTaskSetter}>
      <input type="text" value={inputValue} placeholder="Task ID" onChange={(e) => setInputValue(e.target.value)} />
      <button className="btn-primary" onClick={handleSubmit}>
        Set Task
      </button>
    </div>
  );
}
