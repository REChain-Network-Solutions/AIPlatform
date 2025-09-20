<?php

namespace Tests\YooKassa\Validator\Constraints;

use PHPUnit\Framework\TestCase;
use YooKassa\Validator\Constraints\Url;
use YooKassa\Validator\Constraints\UrlValidator;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\ValidatorParameterException;

class UrlValidatorTest extends TestCase
{
    public function getInstance(): UrlValidator
    {
        return new UrlValidator('className', 'propertyName');
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testValidate($value): void
    {
        $instance = $this->getInstance();
        $this->assertNull($instance->validate($value, new Url));
    }

    public function validDataProvider(): array
    {
        return [
            [null],
            ['http://ya.ru'],
            ['http://user:name@ya.ru'],
            ['http://user:name@ya.ru:8080/test'],
            ['http://user:name@ya.ru:8080/test?query=test'],
            ['http://user:name@ya.ru:8080/test?query=test#test'],
            ['https://xn----8sba6agi1bne8g.xn--p1ai'],
            ['https://xn----8sba6agi1bne8g.xn--p1ai/modules/'],
            ['https://test.xn----8sba6agi1bne8g.xn--p1ai/'],
            ['https://test.test.xn----8sba6agi1bne8g.xn--p1ai/'],
            ['https://www.xn--d1aqf.xn--p1ai/analytics/#']
        ];
    }

    /**
     * @dataProvider invalidDataProvider
     */
    public function testInvalidValidate($value, $expected): void
    {
        $instance = $this->getInstance();
        $this->expectException($expected);
        $instance->validate($value, new Url);
    }

    public function invalidDataProvider(): array
    {
        return [
            [new \stdClass(), ValidatorParameterException::class],
            [[true], ValidatorParameterException::class],
            ['test', InvalidPropertyValueException::class]
        ];
    }
}
