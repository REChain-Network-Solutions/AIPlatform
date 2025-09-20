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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\BankCardProduct;

/**
 * BankCardProductTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class BankCardProductTest extends AbstractTestCase
{
    protected BankCardProduct $object;

    /**
     * @return BankCardProduct
     */
    protected function getTestInstance(): BankCardProduct
    {
        return new BankCardProduct();
    }

    /**
     * @return void
     */
    public function testBankCardProductClassExists(): void
    {
        $this->object = $this->getMockBuilder(BankCardProduct::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(BankCardProduct::class));
        $this->assertInstanceOf(BankCardProduct::class, $this->object);
    }

    /**
     * Test property "code"
     * @dataProvider validCodeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCode($value);
        self::assertNotNull($instance->getCode());
        self::assertNotNull($instance->code);
        self::assertEquals($value, is_array($value) ? $instance->getCode()->toArray() : $instance->getCode());
        self::assertEquals($value, is_array($value) ? $instance->code->toArray() : $instance->code);
    }

    /**
     * Test invalid property "code"
     * @dataProvider invalidCodeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCode($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_code'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_code'));
    }

    /**
     * Test property "name"
     * @dataProvider validNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getName());
        self::assertEmpty($instance->name);
        $instance->setName($value);
        self::assertEquals($value, is_array($value) ? $instance->getName()->toArray() : $instance->getName());
        self::assertEquals($value, is_array($value) ? $instance->name->toArray() : $instance->name);
        if (!empty($value)) {
            self::assertNotNull($instance->getName());
            self::assertNotNull($instance->name);
        }
    }

    /**
     * Test invalid property "name"
     * @dataProvider invalidNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }
}
