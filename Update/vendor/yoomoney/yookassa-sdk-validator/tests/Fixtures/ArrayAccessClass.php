<?php

namespace Tests\YooKassa\Validator\Fixtures;

use ArrayAccess;

class ArrayAccessClass implements ArrayAccess {
    private array $container = [];

    public function __construct()
    {
        $this->container = [
            "one"   => 1,
            "two"   => 2,
            "three" => 3,
        ];
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        if (is_null($offset)) {
            $this->container[] = $value;
        } else {
            $this->container[$offset] = $value;
        }
    }

    public function offsetExists($offset): bool
    {
        return isset($this->container[$offset]);
    }

    public function offsetUnset(mixed $offset): void
    {
        unset($this->container[$offset]);
    }

    public function offsetGet(mixed $offset): mixed
    {
        return $this->container[$offset] ?? null;
    }
}
