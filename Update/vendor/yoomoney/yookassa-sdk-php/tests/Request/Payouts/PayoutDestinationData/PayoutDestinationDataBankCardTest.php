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

namespace Tests\YooKassa\Request\Payouts\PayoutDestinationData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataBankCardCard;

/**
 * PayoutDestinationDataBankCardTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PayoutDestinationDataBankCardTest extends AbstractTestCase
{
    protected PayoutDestinationDataBankCardCard $object;

    /**
     * @return PayoutDestinationDataBankCardCard
     */
    protected function getTestInstance(): PayoutDestinationDataBankCardCard
    {
        return new PayoutDestinationDataBankCardCard();
    }

    /**
     * @return void
     */
    public function testPayoutDestinationDataBankCardCardClassExists(): void
    {
        $this->object = $this->getMockBuilder(PayoutDestinationDataBankCardCard::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PayoutDestinationDataBankCardCard::class));
        $this->assertInstanceOf(PayoutDestinationDataBankCardCard::class, $this->object);
    }

    /**
     * Test property "number"
     * @dataProvider validNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setNumber($value);
        self::assertNotNull($instance->getNumber());
        self::assertNotNull($instance->number);
        self::assertEquals($value, is_array($value) ? $instance->getNumber()->toArray() : $instance->getNumber());
        self::assertEquals($value, is_array($value) ? $instance->number->toArray() : $instance->number);
        self::assertMatchesRegularExpression("/[0-9]{16,19}/", $instance->getNumber());
        self::assertMatchesRegularExpression("/[0-9]{16,19}/", $instance->number);
    }

    /**
     * Test invalid property "number"
     * @dataProvider invalidNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_number'));
    }
}
