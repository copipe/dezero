<p>
<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="400px"
</p>

# DeZero
DeZero is a python package that provides a deep learing framework.  
This package has minimal code and features similar to modern frameworks like PyTorch and TensorFlow.  
This code is based on ["Deep Learning 3 made from scratch"](https://www.oreilly.co.jp/books/9784873119069/), and you can refer to the original source code from [here](https://github.com/oreilly-japan/deep-learning-from-scratch-3).

## Setup

First, launch an ubuntu container using docker.
```
$ cd environments/cpu/
$ docker-compose up -d
```

Once the container has started, connect to it.
```
$ docker exec -it dezero_cpu /bin/bash
```

Next, create a virtual environment for python3.9 with poetry.
```
$ poetry install
```

Setup is complete by activating the virtual environment with the following command.

```
$ poetry shell
```

## Examples
comming soon

## License
DeZero is released under the MIT license.

## References

- [斎藤 康毅. ゼロから作るDeep Learning ❸. O'Reilly Japan. 2020.](https://www.oreilly.co.jp/books/9784873119069/)



<p>
<a href="https://www.amazon.co.jp/dp/4873119065/ref=cm_sw_r_tw_dp_U_x_KiA1Eb39SW14Q"><img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" height="250"></a>
</p>

