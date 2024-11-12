import { Task } from '../../types/types';
import Grid from '../Grid/Grid';
import styles from './styles.module.scss';

export default function TaskExamples({ task }: { task: Task | undefined }) {
  return (
    <div className={[styles.taskExamples, 'hide-scroll'].join(' ')}>
      {task?.train.map((example, index) => (
        <div key={index} className="single-example">
          <div>
            <h5 className="title on-surface-variant body-medium">Input {index}</h5>
            <div className="grid-container">
              <Grid key="input" taskGrid={example.input}></Grid>
            </div>
          </div>

          <div>
            <h5 className="title on-surface-variant body-medium">Output {index}</h5>
            <div className="grid-container">
              <Grid key="output" taskGrid={example.output}></Grid>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
