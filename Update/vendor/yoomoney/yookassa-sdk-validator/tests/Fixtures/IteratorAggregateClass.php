<?php

namespace Tests\YooKassa\Validator\Fixtures;

use IteratorAggregate;
use ArrayIterator;

class IteratorAggregateClass implements IteratorAggregate {
    public object $property;

    public function __construct($value)
    {
        $this->property = $value;
    }

    public function getIterator(): ArrayIterator
    {
        return new ArrayIterator($this);
    }
}