from statistics import mean


def create_report(per_epoch_durations, batch_durations, total_duration):
    print(f"Total duration is {total_duration}")
    print(f"Per epoch duration {mean(per_epoch_durations)}")
    print(f"Per batch duration {mean(batch_durations)}")
