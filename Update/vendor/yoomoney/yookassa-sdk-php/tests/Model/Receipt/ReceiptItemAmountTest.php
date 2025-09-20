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
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use PHPUnit\Framework\TestCase;

/**
 * ReceiptItemAmountTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptItemAmountTest extends AbstractTestCase
{
    protected ReceiptItemAmount $object;

    /**
     * @param mixed|null $value
     * @return ReceiptItemAmount
     */
    protected function getTestInstance(mixed $value = null): ReceiptItemAmount
    {
        return new ReceiptItemAmount($value['value'] ?? null, $value['currency'] ?? null);
    }

    /**
     * @return void
     */
    public function testReceiptItemAmountClassExists(): void
    {
        $this->object = $this->getMockBuilder(ReceiptItemAmount::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ReceiptItemAmount::class));
        $this->assertInstanceOf(ReceiptItemAmount::class, $this->object);
    }

    /**
     * Test property "currency"
     * @dataProvider validCurrencyDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCurrency(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertNotEmpty($instance->getCurrency());
        self::assertNotEmpty($instance->currency);
        $instance->setCurrency($value);
        self::assertNotNull($instance->getCurrency());
        self::assertNotNull($instance->currency);
        self::assertEquals($value, is_array($value) ? $instance->getCurrency()->toArray() : $instance->getCurrency());
        self::assertEquals($value, is_array($value) ? $instance->currency->toArray() : $instance->currency);
        self::assertContains($instance->getCurrency(), ['RUB', 'USD', 'EUR', 'BYN', 'CNY', 'KZT', 'UAH', 'UZS', 'TRY', 'INR', 'MDL', 'AZN', 'AMD']);
        self::assertContains($instance->currency, ['RUB', 'USD', 'EUR', 'BYN', 'CNY', 'KZT', 'UAH', 'UZS', 'TRY', 'INR', 'MDL', 'AZN', 'AMD']);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCurrencyDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_currency'));
    }

    /**
     * Test property "value"
     * @dataProvider validValueDataProvider
     * @param mixed $data
     *
     * @return void
     * @throws Exception
     */
    public function testValue(mixed $data): void
    {
        $instance = $this->getTestInstance();
        self::assertNotEmpty($instance->getValue());
        self::assertNotEmpty($instance->value);
        $instance->setValue($data);
        self::assertNotNull($instance->getValue());
        self::assertNotNull($instance->value);
        self::assertEquals($data, is_array($data) ? $instance->getValue()->toArray() : $instance->getValue());
        self::assertEquals($data, is_array($data) ? $instance->value->toArray() : $instance->value);
        self::assertIsString($instance->getValue());
        self::assertIsString($instance->value);

        $instance->setValue(1);
        self::assertEquals('1.00', $instance->getValue());
        self::assertEquals('1.00', $instance->value);
        self::assertEquals(100, $instance->getIntegerValue());

        $instance->setValue(0.1);
        self::assertEquals('0.10', $instance->getValue());
        self::assertEquals('0.10', $instance->value);
        self::assertEquals(10, $instance->getIntegerValue());

        $instance->setValue(0.01);
        self::assertEquals('0.01', $instance->getValue());
        self::assertEquals('0.01', $instance->value);
        self::assertEquals(1, $instance->getIntegerValue());
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validValueDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_value'));
    }

    /**
     * Test invalid property "value"
     * @dataProvider invalidValueDataProvider
     * @param mixed $data
     * @return void
     */
    public function testInvalidValue(mixed $data): void
    {
        $instance = $this->getTestInstance();

        $this->expectException(InvalidPropertyValueException::class);
        $instance->setValue($data);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidValueDataProvider(): array
    {
        return [
            [-0.1],
            [-1],
            [-100]
        ];
    }

    /**
     * Test valid method "jsonSerialize"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testJsonSerialize(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertEquals($value, $instance->jsonSerialize());

        $value['value'] = [
            'value' => $value['value'] ?? null,
            'currency' => $value['currency'] ?? null
        ];
        $instance = $this->getTestInstance($value);
        self::assertEquals($value['value']['value'], $instance->getValue());
        self::assertEquals($value['currency'], $instance->getCurrency());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }

    /**
     * Test method "increase"
     * @dataProvider validValueDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testIncrease(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->increase($value);
        self::assertNotNull($instance->getValue());
        self::assertNotNull($instance->value);
        self::assertEquals((int) round($value * 100.0), $instance->getIntegerValue());
        self::assertIsString($instance->getValue());
        self::assertIsString($instance->value);
    }

    /**
     * Test invalid method "increase"
     * @dataProvider invalidIncreaseDataProvider
     * @param mixed $data
     * @param string $exceptionClass
     * @return void
     */
    public function testInvalidIncrease(mixed $data, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->increase($data);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIncreaseDataProvider(): array
    {
        return [
            [null, EmptyPropertyValueException::class],
            [-1, InvalidPropertyValueException::class],
        ];
    }

    /**
     * Test method "multiply"
     * @dataProvider validValueDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMultiply(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setValue($value);
        $instance->multiply($value);
        self::assertNotNull($instance->getValue());
        self::assertNotNull($instance->value);
        self::assertEquals((int) round($value * $value), $instance->getValue());
        self::assertIsString($instance->getValue());
        self::assertIsString($instance->value);
    }

    /**
     * Test invalid method "multiply"
     * @dataProvider invalidMultiplyDataProvider
     * @param mixed $data
     * @param string $exceptionClass
     * @return void
     */
    public function testInvalidMultiply(mixed $data, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setValue(0);
        $instance->multiply($data);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMultiplyDataProvider(): array
    {
        return [
            [null, EmptyPropertyValueException::class],
            [-1, InvalidPropertyValueException::class],
            [1, InvalidPropertyValueException::class],
        ];
    }
}
