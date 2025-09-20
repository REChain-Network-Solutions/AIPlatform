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

namespace Tests\YooKassa\Model\Payout;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payout\PayoutDestinationYooMoney;

/**
 * PayoutDestinationYooMoneyTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PayoutDestinationYooMoneyTest extends AbstractTestCase
{
    protected PayoutDestinationYooMoney $object;

    /**
     * @return PayoutDestinationYooMoney
     */
    protected function getTestInstance(): PayoutDestinationYooMoney
    {
        return new PayoutDestinationYooMoney();
    }

    /**
     * @return void
     */
    public function testPayoutDestinationYooMoneyClassExists(): void
    {
        $this->object = $this->getMockBuilder(PayoutDestinationYooMoney::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PayoutDestinationYooMoney::class));
        $this->assertInstanceOf(PayoutDestinationYooMoney::class, $this->object);
    }

    /**
     * Test property "type"
     *
     * @return void
     * @throws Exception
     */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertContains($instance->getType(), ['yoo_money']);
        self::assertContains($instance->type, ['yoo_money']);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
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
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
     * Test property "account_number"
     * @dataProvider validAccountNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAccountNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAccountNumber($value);
        self::assertNotNull($instance->getAccountNumber());
        self::assertNotNull($instance->account_number);
        self::assertEquals($value, is_array($value) ? $instance->getAccountNumber()->toArray() : $instance->getAccountNumber());
        self::assertEquals($value, is_array($value) ? $instance->account_number->toArray() : $instance->account_number);
        self::assertLessThanOrEqual(33, is_string($instance->getAccountNumber()) ? mb_strlen($instance->getAccountNumber()) : $instance->getAccountNumber());
        self::assertLessThanOrEqual(33, is_string($instance->account_number) ? mb_strlen($instance->account_number) : $instance->account_number);
        self::assertGreaterThanOrEqual(11, is_string($instance->getAccountNumber()) ? mb_strlen($instance->getAccountNumber()) : $instance->getAccountNumber());
        self::assertGreaterThanOrEqual(11, is_string($instance->account_number) ? mb_strlen($instance->account_number) : $instance->account_number);
    }

    /**
     * Test invalid property "account_number"
     * @dataProvider invalidAccountNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAccountNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAccountNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_number'));
    }
}
