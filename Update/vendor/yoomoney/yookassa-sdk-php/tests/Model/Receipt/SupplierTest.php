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
use YooKassa\Model\Receipt\Supplier;

/**
 * ReceiptItemSupplierTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SupplierTest extends AbstractTestCase
{
    protected Supplier $object;

    /**
     * @return Supplier
     */
    protected function getTestInstance(): Supplier
    {
        return new Supplier();
    }

    /**
     * @return void
     */
    public function testReceiptItemSupplierClassExists(): void
    {
        $this->object = $this->getMockBuilder(Supplier::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Supplier::class));
        $this->assertInstanceOf(Supplier::class, $this->object);
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
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->getPhone());
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->phone);
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
            self::assertMatchesRegularExpression("/\\d{10}|\\d{12}/", $instance->getInn());
            self::assertMatchesRegularExpression("/\\d{10}|\\d{12}/", $instance->inn);
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
}
