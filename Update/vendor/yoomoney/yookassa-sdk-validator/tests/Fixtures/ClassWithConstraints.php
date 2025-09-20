<?php

namespace Tests\YooKassa\Validator\Fixtures;

use YooKassa\Validator\Constraints as Assert;

class ClassWithConstraints
{
    #[Assert\NotNull]
    private ?string $prop = null;

    public function __construct(?string $propVal = null)
    {
        $this->prop = $propVal;
    }
}