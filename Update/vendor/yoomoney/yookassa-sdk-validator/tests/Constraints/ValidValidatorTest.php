<?php

namespace Tests\YooKassa\Validator\Constraints;

use Tests\YooKassa\Validator\Fixtures\ClassWithConstraints;
use Tests\YooKassa\Validator\Fixtures\IteratorAggregateClass;
use YooKassa\Validator\Constraints\Valid;
use YooKassa\Validator\Constraints\ValidValidator;
use PHPUnit\Framework\TestCase;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\ValidatorParameterException;

class ValidValidatorTest extends TestCase
{
    private function getInstance(): ValidValidator
    {
        return new ValidValidator('className', 'propertyName');
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testValidate($value)
    {
        $instance = $this->getInstance();
        if ($value !== null) {
            $this->expectException(EmptyPropertyValueException::class);
            $instance->validate($value, new Valid);
        } else {
            $this->assertNull($instance->validate($value, new Valid));
        }

    }

    public function validDataProvider(): array
    {
        return [
            [null],
            [new IteratorAggregateClass(new ClassWithConstraints())],
            [[new ClassWithConstraints]],
            [new ClassWithConstraints]
        ];
    }

    /**
     * @dataProvider invalidDataProvider
     */
    public function testInvalidValidate($value)
    {
        $instance = $this->getInstance();
        $this->expectException(ValidatorParameterException::class);
        $instance->validate($value, new Valid);
    }

    public function invalidDataProvider(): array
    {
        return [
            [123],
            [123.123],
            ['test']
        ];
    }
}