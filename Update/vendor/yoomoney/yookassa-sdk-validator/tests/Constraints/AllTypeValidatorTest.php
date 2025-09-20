<?php

namespace Tests\YooKassa\Validator\Constraints;

use PHPUnit\Framework\TestCase;
use stdClass;
use Tests\YooKassa\Validator\Fixtures\ArrayAccessClass;
use Tests\YooKassa\Validator\Fixtures\ClassWithConstraints;
use Tests\YooKassa\Validator\Fixtures\IteratorAggregateClass;
use YooKassa\Validator\Constraints\AllType;
use YooKassa\Validator\Constraints\AllTypeValidator;
use YooKassa\Validator\Exceptions\ValidatorParameterException;

class AllTypeValidatorTest extends TestCase
{
    private function getInstance(): AllTypeValidator
    {
        return new AllTypeValidator('className', 'propertyName');
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testValidate($value)
    {
        $instance = $this->getInstance();
        $this->assertNull($instance->validate($value, new AllType(ClassWithConstraints::class)));
    }

    public function validDataProvider()
    {
        return [
            ['value' => []],
            ['value' => null],
            ['value' => new IteratorAggregateClass(new ClassWithConstraints)]
        ];
    }

    /**
     * @dataProvider invalidDataProvider
     */
    public function testInvalidValidate($value)
    {
        $instance = $this->getInstance();
        $this->expectException(ValidatorParameterException::class);
        $instance->validate($value, new AllType('string'));
    }

    public function invalidDataProvider(): array
    {
        return [
            ['value' => new ArrayAccessClass],
            ['value' => 123],
            ['value' => 'test'],
            ['value' => 123.123],
            ['value' => new stdClass()]
        ];
    }
}
