from hydra.experimental import compose, initialize

from flower_classifier.train import train


def test_hydra_config(tmpdir, trainer="test", dataset="random"):
    cmd_line = "trainer={} dataset={} hydra.run.dir={}"
    with initialize(config_path="../../conf", job_name="cofig"):
        cfg = compose(
            config_name="config",
            overrides=cmd_line.format(trainer, dataset, tmpdir).split(" "),
        )
        train(cfg)
