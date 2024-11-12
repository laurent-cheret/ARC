import { forwardRef } from 'react';
import { StepOutput, TaskGrid } from '../../types/types';
import Grid from '../Grid/Grid';
import styles from './styles.module.scss';

const MemoryStep = forwardRef<HTMLDivElement, { stepOutput: StepOutput }>(({ stepOutput }, ref) => {
  const isMemoryEmpty = stepOutput.memory_grids.every((slot) => slot.length == 0);

  return (
    <div className={styles.memOutput} ref={ref}>
      {isMemoryEmpty && (
        <div className="memory-empty">
          <div className="on-surface-variant body-small">Empty</div>
        </div>
      )}

      {!isMemoryEmpty && (
        <div className="examples-container hide-scroll">
          {stepOutput.memory_grids.map((gridsList: TaskGrid[], index: number) => (
            <>
              {gridsList.length > 0 && (
                <div className="grid-column" key={index}>
                  <>
                    {gridsList.map((grid: TaskGrid, index2: number) => (
                      <div className="grid-container">
                        <Grid taskGrid={grid} key={index2}></Grid>
                      </div>
                    ))}
                  </>
                </div>
              )}
            </>
          ))}
        </div>
      )}
    </div>
  );
});

export default MemoryStep;
