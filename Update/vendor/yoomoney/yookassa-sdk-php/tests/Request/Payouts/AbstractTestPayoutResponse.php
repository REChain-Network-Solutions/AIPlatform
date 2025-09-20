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

use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Payment\PaymentMethod\BankCardType;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payout\Payout;
use YooKassa\Model\Payout\PayoutCancellationDetailsPartyCode;
use YooKassa\Model\Payout\PayoutCancellationDetailsReasonCode;
use YooKassa\Model\Payout\PayoutDestinationType;
use YooKassa\Model\Payout\PayoutInterface;
use YooKassa\Model\Payout\PayoutStatus;

/**
 * AbstractTestPayoutResponse
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestPayoutResponse extends TestCase
{
    /**
     * @dataProvider validDataProvider
     */
    public function testGetId(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['id'], $instance->getId());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetAmount(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals(number_format($options['amount']['value'], 2, '.', ''), $instance->getAmount()->getValue());
        self::assertEquals($options['amount']['currency'], $instance->getAmount()->getCurrency());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetStatus(array $options): void
    {
        $instance = $this->getTestInstance($options);
        self::assertEquals($options['status'], $instance->getStatus());
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetPayoutDestination(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['payout_destination'])) {
            self::assertNull($instance->getPayoutDestination());
            self::assertNull($instance->payout_destination);
        } else {
            self::assertEquals($options['payout_destination'], $instance->getPayoutDestination()->toArray());
            self::assertEquals($options['payout_destination'], $instance->payoutDestination->toArray());
            self::assertEquals($options['payout_destination'], $instance->payout_destination->toArray());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetCreatedAt(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['created_at'])) {
            self::assertNull($instance->getCreatedAt());
            self::assertNull($instance->created_at);
            self::assertNull($instance->createdAt);
        } else {
            self::assertEquals($options['created_at'], $instance->getCreatedAt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetSucceededAt(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['succeeded_at'])) {
            self::assertNull($instance->getSucceededAt());
            self::assertNull($instance->succeeded_at);
            self::assertNull($instance->succeededAt);
        } else {
            self::assertEquals($options['succeeded_at'], $instance->getSucceededAt()->format(YOOKASSA_DATE));
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetDeal(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['deal'])) {
            self::assertNull($instance->getDeal());
            self::assertNull($instance->deal);
        } else {
            $instance->setDeal($options['deal']);
            self::assertSame($options['deal'], $instance->getDeal()->toArray());
            self::assertSame($options['deal'], $instance->deal->toArray());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetTest(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (!isset($options['test'])) {
            self::assertNull($instance->getTest());
        } else {
            self::assertEquals($options['test'], $instance->getTest());
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetCancellationDetails(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['cancellation_details'])) {
            self::assertNull($instance->getCancellationDetails());
        } else {
            self::assertEquals(
                $options['cancellation_details']['party'],
                $instance->getCancellationDetails()->getParty()
            );
            self::assertEquals(
                $options['cancellation_details']['reason'],
                $instance->getCancellationDetails()->getReason()
            );
        }
    }

    /**
     * @dataProvider validDataProvider
     */
    public function testGetMetadata(array $options): void
    {
        $instance = $this->getTestInstance($options);
        if (empty($options['metadata'])) {
            self::assertNull($instance->getMetadata());
        } else {
            self::assertEquals($options['metadata'], $instance->getMetadata()->toArray());
        }
    }

    public static function validDataProvider(): array
    {
        $result = [];
        $cancellationDetailsParties = PayoutCancellationDetailsPartyCode::getValidValues();
        $countCancellationDetailsParties = count($cancellationDetailsParties);
        $cancellationDetailsReasons = PayoutCancellationDetailsReasonCode::getValidValues();
        $countCancellationDetailsReasons = count($cancellationDetailsReasons);
        $payoutDestinations = [
            PayoutDestinationType::YOO_MONEY => [
                'type' => PaymentMethodType::YOO_MONEY,
                'account_number' => Random::str(11, 33, '1234567890'),
            ],
            PayoutDestinationType::BANK_CARD => [
                'type' => PaymentMethodType::BANK_CARD,
                'card' => [
                    'first6' => Random::str(6, '0123456789'),
                    'last4' => Random::str(4, '0123456789'),
                    'card_type' => Random::value(BankCardType::getValidValues()),
                    'issuer_country' => 'RU',
                    'issuer_name' => 'SberBank',
                ],
            ],
            PayoutDestinationType::SBP => [
                'type' => PaymentMethodType::SBP,
                'phone' => Random::str(4, 15, '0123456789'),
                'bank_id' => Random::str(12, 12, '0123456789'),
                'recipient_checked' => Random::bool(),
            ]
        ];

        $result[] = [
            [
                'id' => Random::str(36, 50),
                'status' => Random::value(PayoutStatus::getValidValues()),
                'amount' => ['value' => Random::int(1, 10000), 'currency' => 'RUB'],
                'description' => Random::str(1, Payout::MAX_LENGTH_DESCRIPTION),
                'payout_destination' => $payoutDestinations[Random::value(PayoutDestinationType::getValidValues())],
                'created_at' => date(YOOKASSA_DATE, Random::int(111111111, time())),
                'succeeded_at' => date(YOOKASSA_DATE, Random::int(111111111, time())),
                'test' => true,
                'deal' => ['id' => Random::str(36, 50)],
                'metadata' => ['order_id' => '37'],
                'cancellation_details' => [
                    'party' => Random::value($cancellationDetailsParties),
                    'reason' => Random::value($cancellationDetailsReasons),
                ],
            ],
        ];
        $result[] = [
            [
                'id' => Random::str(36, 50),
                'status' => Random::value(PayoutStatus::getValidValues()),
                'amount' => ['value' => Random::int(1, 10000), 'currency' => 'RUB'],
                'description' => Random::str(1, Payout::MAX_LENGTH_DESCRIPTION),
                'payout_destination' => $payoutDestinations[Random::value(PayoutDestinationType::getValidValues())],
                'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'succeeded_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'test' => true,
                'metadata' => null,
                'cancellation_details' => null,
            ],
        ];

        for ($i = 0; $i < 20; $i++) {
            $payment = [
                'id' => Random::str(36, 50),
                'status' => Random::value(PayoutStatus::getValidValues()),
                'amount' => ['value' => Random::int(1, 10000), 'currency' => 'RUB'],
                'description' => (0 === $i ? null : (1 === $i ? '' : (2 === $i ? Random::str(Payout::MAX_LENGTH_DESCRIPTION)
                    : Random::str(1, Payout::MAX_LENGTH_DESCRIPTION)))),
                'payout_destination' => $payoutDestinations[Random::value(PayoutDestinationType::getValidValues())],
                'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'succeeded_at' => date(YOOKASSA_DATE, Random::int(1, time())),
                'test' => (bool) ($i % 2),
                'metadata' => [Random::str(3, 128, 'abcdefghijklmnopqrstuvwxyz') => Random::str(1, 512)],
                'cancellation_details' => [
                    'party' => $cancellationDetailsParties[$i % $countCancellationDetailsParties],
                    'reason' => $cancellationDetailsReasons[$i % $countCancellationDetailsReasons],
                ],
            ];
            $result[] = [$payment];
        }

        return $result;
    }

    /**
     * @param mixed $options
     */
    abstract protected function getTestInstance(mixed $options): PayoutInterface;
}
