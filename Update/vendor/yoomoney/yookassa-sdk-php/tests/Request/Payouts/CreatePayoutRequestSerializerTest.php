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

namespace Tests\YooKassa\Request\Payouts;

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Request\Payouts\CreatePayoutRequest;
use YooKassa\Request\Payouts\CreatePayoutRequestSerializer;
use YooKassa\Request\Payouts\IncomeReceiptData;
use YooKassa\Request\Payouts\PayoutPersonalData;
use YooKassa\Request\Payouts\PayoutSelfEmployedInfo;

/**
 * CreatePayoutRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePayoutRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new CreatePayoutRequestSerializer();
        $instance = CreatePayoutRequest::builder()->build($options);
        $data = $serializer->serialize($instance);

        $request = new CreatePayoutRequest($options);
        $expected = $request->toArray();

        self::assertEquals($expected, $data);
    }

    /**
     * @throws Exception
     */
    public static function validDataProvider(): array
    {
        $metadata = new Metadata();
        $metadata->test = 'test';
        $result = [
            [
                [
                    'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                    'payoutToken' => uniqid('', true),
                    'payoutDestinationData' => null,
                    'metadata' => null,
                    'description' => null,
                    'deal' => null,
                    'payment_method_id' => null,
                    'self_employed' => null,
                    'receipt_data' => null,
                    'personal_data' => null,
                ],
            ],
            [
                [
                    'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                    'payoutToken' => '',
                    'payoutDestinationData' => Random::value(self::payoutDestinationData()),
                    'metadata' => [Random::str(1)],
                    'description' => '',
                    'deal' => new PayoutDealInfo(['id' => Random::str(36, 50)]),
                    'payment_method_id' => '',
                    'self_employed' => new PayoutSelfEmployedInfo(['id' => Random::str(36, 50)]),
                    'receipt_data' => new IncomeReceiptData(['service_name' => Random::str(1, 50)]),
                    'personal_data' => [new PayoutPersonalData(['id' => Random::str(36, 50)])],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $even = ($i % 3);
            $request = [
                'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                'payoutToken' => $even === 0 ? uniqid('', true) : null,
                'payoutDestinationData' => $even === 1 ? Random::value(self::payoutDestinationData()) : null,
                'metadata' => $even ? $metadata : ['test' => 'test'],
                'description' => Random::str(5, 128),
                'deal' => $even ? new PayoutDealInfo(['id' => Random::str(36, 50)]) : ['id' => Random::str(36, 50)],
                'payment_method_id' => $even === 2 ? Random::str(5, 128) : null,
                'self_employed' => $even ? new PayoutSelfEmployedInfo(['id' => Random::str(36, 50)]) : ['id' => Random::str(36, 50)],
                'receipt_data' => $even ? new IncomeReceiptData(['service_name' => Random::str(36, 50), 'amount' => new MonetaryAmount(Random::int(1, 1000000))]) : ['service_name' => Random::str(36, 50), 'amount' => ['value' => Random::int(1, 1000000) . '.00', 'currency' => CurrencyCode::RUB]],
                'personal_data' => [$even ? new PayoutPersonalData(['id' => Random::str(36, 50)]) : ['id' => Random::str(36, 50)]],
            ];
            $result[] = [$request];
        }

        return $result;
    }

    public static function payoutDestinationData(): array
    {
        return [
            [
                'type' => PaymentMethodType::YOO_MONEY,
                'account_number' => Random::str(11, 33, '0123456789'),
            ],
            [
                'type' => PaymentMethodType::BANK_CARD,
                'card' => [
                    'number' => Random::str(16, 16, '0123456789'),
                ],
            ],
            [
                'type' => PaymentMethodType::SBP,
                'phone' => Random::str(4, 15, '0123456789'),
                'bank_id' => Random::str(12, 12, '0123456789'),
            ],
        ];
    }
}
