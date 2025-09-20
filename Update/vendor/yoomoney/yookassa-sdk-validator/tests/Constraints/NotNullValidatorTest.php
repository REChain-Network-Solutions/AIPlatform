<?php

namespace Tests\YooKassa\Validator\Constraints;

use YooKassa\Validator\Constraints\NotNull;
use YooKassa\Validator\Constraints\NotNullValidator;
use PHPUnit\Framework\TestCase;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;

class NotNullValidatorTest extends TestCase
{
    public function getInstance()
    {
        return new NotNullValidator('className', 'propertyName');
    }

    public function testValidate()
    {
        $constraint = new NotNull();
        $instance = $this->getInstance();
        $this->assertNull($instance->validate('NotNull', $constraint));
        $this->expectException(EmptyPropertyValueException::class);
        $instance->validate(null, $constraint);
    }
}
