from hydra.experimental import compose, initialize

from flower_classifier.train import train


def test_hydra_config(tmpdir, trainer="test", dataset="random", transforms="normalize"):
    cmd_line = "trainer={} dataset={} transforms={} hydra.run.dir={}"
    with initialize(config_path="../../conf"):
        cfg = compose(
            config_name="config",
            overrides=cmd_line.format(trainer, dataset, transforms, tmpdir).split(" "),
        )
        train(cfg)
