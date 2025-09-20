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

namespace Tests\YooKassa\Request\Payments\PaymentData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Helpers\Random;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentData\PaymentDataBankCard;
use YooKassa\Request\Payments\PaymentData\PaymentDataBankCardCard;

/**
 * PaymentDataBankCardTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentDataBankCardTest extends AbstractTestCase
{
    protected PaymentDataBankCard $object;

    /**
     * @return PaymentDataBankCard
     */
    protected function getTestInstance(): PaymentDataBankCard
    {
        return new PaymentDataBankCard();
    }

    /**
     * @return void
     */
    public function testPaymentDataBankCardClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentDataBankCard::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentDataBankCard::class));
        $this->assertInstanceOf(PaymentDataBankCard::class, $this->object);
    }

    /**
     * Test property "card"
     * @dataProvider validCardDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCard(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCard());
        self::assertEmpty($instance->card);
        $instance->setCard($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getCard());
            self::assertNotNull($instance->card);
        } else {
            self::assertEquals($value, is_array($value) ? $instance->getCard()->toArray() : $instance->getCard());
            self::assertEquals($value, is_array($value) ? $instance->card->toArray() : $instance->card);
        }
    }

    /**
     * Test invalid property "card"
     * @dataProvider invalidCardDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCard(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCard($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardDataProvider(): array
    {
        return [
            [null],
            [new PaymentDataBankCardCard([
                'number' => Random::str(16, '0123456789'),
                'expiry_year' => (string) Random::int(2023, 2045),
                'expiry_month' => str_pad((string) Random::int(1, 12), 2, '0', STR_PAD_LEFT),
                'cardholder' => Random::str(1, 26, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ \'-'),
            ])],
            [''],
            [[
                'number' => Random::str(16, '0123456789'),
                'expiry_year' => (string) Random::int(2023, 2030),
                'expiry_month' => str_pad((string) Random::int(1, 12), 2, '0', STR_PAD_LEFT),
                'cardholder' => Random::str(1, 26, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ \'-'),
            ]],
        ];
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_card'));
    }
}
