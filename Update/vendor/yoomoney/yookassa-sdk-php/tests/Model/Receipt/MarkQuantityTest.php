<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Model\Receipt;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Receipt\MarkQuantity;

/**
 * MarkQuantityTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class MarkQuantityTest extends AbstractTestCase
{
    protected MarkQuantity $object;

    /**
     * @return MarkQuantity
     */
    protected function getTestInstance(): MarkQuantity
    {
        return new MarkQuantity();
    }

    /**
     * @return void
     */
    public function testMarkQuantityClassExists(): void
    {
        $this->object = $this->getMockBuilder(MarkQuantity::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(MarkQuantity::class));
        $this->assertInstanceOf(MarkQuantity::class, $this->object);
    }

    /**
     * Test property "numerator"
     * @dataProvider validNumeratorDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testNumerator(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setNumerator($value);
        self::assertNotNull($instance->getNumerator());
        self::assertNotNull($instance->numerator);
        self::assertEquals($value, is_array($value) ? $instance->getNumerator()->toArray() : $instance->getNumerator());
        self::assertEquals($value, is_array($value) ? $instance->numerator->toArray() : $instance->numerator);
        self::assertGreaterThanOrEqual(1, is_string($instance->getNumerator()) ? mb_strlen($instance->getNumerator()) : $instance->getNumerator());
        self::assertGreaterThanOrEqual(1, is_string($instance->numerator) ? mb_strlen($instance->numerator) : $instance->numerator);
        self::assertIsNumeric($instance->getNumerator());
        self::assertIsNumeric($instance->numerator);
    }

    /**
     * Test invalid property "numerator"
     * @dataProvider invalidNumeratorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidNumerator(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setNumerator($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validNumeratorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_numerator'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidNumeratorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_numerator'));
    }

    /**
     * Test property "denominator"
     * @dataProvider validDenominatorDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDenominator(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setDenominator($value);
        self::assertNotNull($instance->getDenominator());
        self::assertNotNull($instance->denominator);
        self::assertEquals($value, is_array($value) ? $instance->getDenominator()->toArray() : $instance->getDenominator());
        self::assertEquals($value, is_array($value) ? $instance->denominator->toArray() : $instance->denominator);
        self::assertGreaterThanOrEqual(1, is_string($instance->getDenominator()) ? mb_strlen($instance->getDenominator()) : $instance->getDenominator());
        self::assertGreaterThanOrEqual(1, is_string($instance->denominator) ? mb_strlen($instance->denominator) : $instance->denominator);
        self::assertIsNumeric($instance->getDenominator());
        self::assertIsNumeric($instance->denominator);
    }

    /**
     * Test invalid property "denominator"
     * @dataProvider invalidDenominatorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDenominator(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDenominator($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDenominatorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_denominator'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDenominatorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_denominator'));
    }
}
