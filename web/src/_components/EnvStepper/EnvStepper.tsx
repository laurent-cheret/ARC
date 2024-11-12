import { useEffect, useRef } from 'react';
import { StepOutput } from '../../types/types';
import EnvStep from '../EnvStep/EnvStep';
import styles from './styles.module.scss';
import MemoryStep from '../MemoryStep/MemoryStep';

export default function EnvStepper({ loadingDemo, stepOutputs }: { loadingDemo: boolean; stepOutputs: StepOutput[] }) {
  const isOutputsEmpty = stepOutputs.length == 0;

  const itemsRef = useRef<Map<number, HTMLDivElement> | null>(null);
  const memItemsRef = useRef<Map<number, HTMLDivElement> | null>(null);
  const scrollIndexRef = useRef(0);

  useEffect(() => {
    scrollIndexRef.current = 0;
  }, [stepOutputs]);

  function getMap(): Map<number, HTMLDivElement> {
    if (!itemsRef.current) {
      itemsRef.current = new Map<number, HTMLDivElement>();
    }
    return itemsRef.current;
  }

  function getMapMem(): Map<number, HTMLDivElement> {
    if (!memItemsRef.current) {
      memItemsRef.current = new Map<number, HTMLDivElement>();
    }
    return memItemsRef.current;
  }

  function scrollToElm(index: number) {
    const map = getMap();
    const node = map.get(index) as HTMLDivElement;
    node.scrollIntoView({
      // behavior: 'smooth',
      behavior: 'instant',
      block: 'nearest',
      inline: 'center',
    });

    const mapMem = getMapMem();
    const nodeMem = mapMem.get(index) as HTMLDivElement;
    nodeMem.scrollIntoView({
      behavior: 'instant',
      block: 'nearest',
      inline: 'center',
    });
  }

  const handleScrollTo = (index: number) => {
    index = Math.max(index, 0);
    index = Math.min(index, stepOutputs.length - 1);
    scrollToElm(index);
    scrollIndexRef.current = index;
  };

  return (
    <>
      {loadingDemo && <div className={styles.skeleton}></div>}

      {!loadingDemo && (
        <div className={styles.stepper}>
          <div className={styles.envStepper}>
            <div className="steps-container">
              {stepOutputs.length == 0 && <img src="/empty_stepper.png" alt="empty stepper" />}

              {stepOutputs.map((stepOutput, index) => (
                <EnvStep
                  key={index}
                  stepOutput={stepOutput}
                  ref={(node) => {
                    const map = getMap();
                    if (node) {
                      map.set(index, node);
                    } else {
                      map.delete(index);
                    }
                  }}
                ></EnvStep>
              ))}
            </div>

            <div className="commands">
              <button className="btn-primary" disabled={isOutputsEmpty} onClick={() => handleScrollTo(0)}>
                &lt;&lt;
              </button>

              <button
                className="btn-primary"
                disabled={isOutputsEmpty}
                onClick={() => handleScrollTo(scrollIndexRef.current - 1)}
              >
                Previous
              </button>
              <button
                className="btn-primary"
                disabled={isOutputsEmpty}
                onClick={() => handleScrollTo(scrollIndexRef.current + 1)}
              >
                Next
              </button>

              <button
                className="btn-primary"
                disabled={isOutputsEmpty}
                onClick={() => handleScrollTo(stepOutputs.length - 1)}
              >
                &gt;&gt;
              </button>
            </div>
          </div>

          <div className="right-side">
            <h5 className="title-mem headline-small on-surface-variant">Memory state</h5>

            <div className="memory-aside">
              {stepOutputs.map((stepOutput, index) => (
                <MemoryStep
                  key={`mem-${index}`}
                  stepOutput={stepOutput}
                  ref={(node) => {
                    const mapMem = getMapMem();
                    if (node) {
                      mapMem.set(index, node);
                    } else {
                      mapMem.delete(index);
                    }
                  }}
                ></MemoryStep>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
