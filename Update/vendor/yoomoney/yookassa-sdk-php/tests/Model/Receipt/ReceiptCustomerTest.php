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
use YooKassa\Model\Receipt\ReceiptCustomer;

/**
 * ReceiptCustomerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptCustomerTest extends AbstractTestCase
{
    protected ReceiptCustomer $object;

    /**
     * @return ReceiptCustomer
     */
    protected function getTestInstance(): ReceiptCustomer
    {
        return new ReceiptCustomer();
    }

    /**
     * @return void
     */
    public function testReceiptCustomerClassExists(): void
    {
        $this->object = $this->getMockBuilder(ReceiptCustomer::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ReceiptCustomer::class));
        $this->assertInstanceOf(ReceiptCustomer::class, $this->object);
    }

    /**
     * Test property "full_name"
     * @dataProvider validFullNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFullName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getFullName());
        self::assertEmpty($instance->full_name);
        self::assertIsBool($instance->isEmpty());
        $instance->setFullName($value);
        self::assertEquals($value, is_array($value) ? $instance->getFullName()->toArray() : $instance->getFullName());
        self::assertEquals($value, is_array($value) ? $instance->full_name->toArray() : $instance->full_name);
        if (!empty($value)) {
            self::assertNotNull($instance->getFullName());
            self::assertNotNull($instance->full_name);
        }
    }

    /**
     * Test invalid property "full_name"
     * @dataProvider invalidFullNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFullName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFullName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFullNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFullNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_full_name'));
    }

    /**
     * Test property "inn"
     * @dataProvider validInnDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testInn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getInn());
        self::assertEmpty($instance->inn);
        $instance->setInn($value);
        self::assertEquals($value, is_array($value) ? $instance->getInn()->toArray() : $instance->getInn());
        self::assertEquals($value, is_array($value) ? $instance->inn->toArray() : $instance->inn);
        if (!empty($value)) {
            self::assertNotNull($instance->getInn());
            self::assertNotNull($instance->inn);
        }
    }

    /**
     * Test invalid property "inn"
     * @dataProvider invalidInnDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidInn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setInn($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
     * Test property "email"
     * @dataProvider validEmailDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEmail(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEmail());
        self::assertEmpty($instance->email);
        $instance->setEmail($value);
        self::assertEquals($value, is_array($value) ? $instance->getEmail()->toArray() : $instance->getEmail());
        self::assertEquals($value, is_array($value) ? $instance->email->toArray() : $instance->email);
        if (!empty($value)) {
            self::assertNotNull($instance->getEmail());
            self::assertNotNull($instance->email);
        }
    }

    /**
     * Test invalid property "email"
     * @dataProvider invalidEmailDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEmail(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEmail($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEmailDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_email'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEmailDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_email'));
    }

    /**
     * Test property "phone"
     * @dataProvider validPhoneDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPhone(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPhone());
        self::assertEmpty($instance->phone);
        $instance->setPhone($value);
        self::assertEquals($value, is_array($value) ? $instance->getPhone()->toArray() : $instance->getPhone());
        self::assertEquals($value, is_array($value) ? $instance->phone->toArray() : $instance->phone);
        if (!empty($value)) {
            self::assertNotNull($instance->getPhone());
            self::assertNotNull($instance->phone);
        }
    }

    /**
     * Test invalid property "phone"
     * @dataProvider invalidPhoneDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPhone(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPhone($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
    }
}
