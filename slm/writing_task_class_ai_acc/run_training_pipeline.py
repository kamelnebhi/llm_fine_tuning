import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n🚀 Starting {script_name} ...\n")
    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, script_name],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with error code {e.returncode}")
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\n✅ {script_name} finished successfully in {duration/60:.2f} minutes.\n")


def main():
    print("=" * 60)
    print("SEQUENTIAL TRAINING PIPELINE STARTED")
    print("=" * 60)

    run_script("roberta_trainer_cross_encoder.py")

    # Optional small pause (good practice for GPU cleanup)
    print("⏳ Waiting 10 seconds before starting next training...")
    time.sleep(10)

    run_script("roberta_trainer_cross_encoder_multitask.py")

    print("=" * 60)
    print("🎉 ALL TRAININGS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()