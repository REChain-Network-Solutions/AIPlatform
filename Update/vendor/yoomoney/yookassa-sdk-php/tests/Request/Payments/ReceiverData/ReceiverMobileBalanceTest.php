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

namespace Tests\YooKassa\Request\Payments\ReceiverData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\ReceiverData\ReceiverMobileBalance;

/**
 * ReceiverMobileBalanceTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ReceiverMobileBalanceTest extends AbstractTestCase
{
    protected ReceiverMobileBalance $object;

    /**
    * @return ReceiverMobileBalance
    */
    protected function getTestInstance(): ReceiverMobileBalance
    {
        return new ReceiverMobileBalance();
    }

    /**
    * @return void
    */
    public function testReceiverMobileBalanceClassExists(): void
    {
        $this->object = $this->getMockBuilder(ReceiverMobileBalance::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ReceiverMobileBalance::class));
        $this->assertInstanceOf(ReceiverMobileBalance::class, $this->object);
    }

    /**
     * Test property "type"
     * @dataProvider validTypeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertContains($instance->getType(), ['mobile_balance']);
        self::assertContains($instance->type, ['mobile_balance']);
    }

    /**
    * Test invalid property "type"
    * @dataProvider invalidTypeDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
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
        $instance->setPhone($value);
        self::assertNotNull($instance->getPhone());
        self::assertNotNull($instance->phone);
        self::assertEquals($value, $instance->getPhone());
        self::assertEquals($value, $instance->phone);
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
