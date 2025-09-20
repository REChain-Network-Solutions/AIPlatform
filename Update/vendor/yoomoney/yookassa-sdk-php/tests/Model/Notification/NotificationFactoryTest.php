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

namespace Tests\YooKassa\Model\Notification;

use Exception;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use stdClass;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\DealStatus;
use YooKassa\Model\Deal\DealType;
use YooKassa\Model\Deal\FeeMoment;
use YooKassa\Model\Notification\AbstractNotification;
use YooKassa\Model\Notification\NotificationEventType;
use YooKassa\Model\Notification\NotificationFactory;
use YooKassa\Model\Notification\NotificationType;
use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Model\Payment\PaymentMethod\BankCardType;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payment\PaymentStatus;
use YooKassa\Model\Payment\ReceiptRegistrationStatus;
use YooKassa\Model\Payout\Payout;
use YooKassa\Model\Payout\PayoutCancellationDetailsPartyCode;
use YooKassa\Model\Payout\PayoutCancellationDetailsReasonCode;
use YooKassa\Model\Payout\PayoutStatus;
use YooKassa\Model\Refund\RefundStatus;
use YooKassa\Request\Deals\DealResponse;
use YooKassa\Request\Payments\PaymentResponse;
use YooKassa\Request\Payouts\PayoutResponse;
use YooKassa\Request\Refunds\RefundResponse;

if (!defined('YOOKASSA_DATE')) {
    define('YOOKASSA_DATE', 'Y-m-d\\TH:i:s.v\\Z');
}

/**
 * NotificationFactoryTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class NotificationFactoryTest extends TestCase
{
    /**
     * @dataProvider invalidDataArrayDataProvider
     *
     * @param mixed $options
     */
    public function testInvalidFactory(mixed $options): void
    {
        $this->expectException(InvalidArgumentException::class);
        $instance = $this->getTestInstance();
        $instance->factory($options);
    }

    /**
     * @dataProvider validArrayDataProvider
     */
    public function testFactory(array $options): void
    {
        $instance = $this->getTestInstance();
        $event = $options['event'];
        $notification = $instance->factory($options);
        self::assertNotNull($notification);
        self::assertInstanceOf(AbstractNotification::class, $notification);

        self::assertEquals($event, $notification->getEvent());
        foreach ($options as $property => $value) {
            if ('object' !== $property) {
                self::assertEquals($notification->{$property}, $value);
            } else {
                $this->assertObject($event, $notification->{$property}, $value);
            }
        }
    }

    /**
     * @throws Exception
     */
    public function validArrayDataProvider(): array
    {
        $result = [];

        for ($i = 0; $i < 12; $i++) {
            $eventType = Random::value(NotificationEventType::getEnabledValues());

            switch ($eventType) {
                case NotificationEventType::REFUND_SUCCEEDED:
                    $notification = $this->getRefundNotification();

                    break;

                case NotificationEventType::DEAL_CLOSED:
                    $notification = $this->getDealNotification();

                    break;

                case NotificationEventType::PAYOUT_SUCCEEDED:
                case NotificationEventType::PAYOUT_CANCELED:
                    $notification = $this->getPayoutNotification($eventType);

                    break;

                default:
                    $notification = $this->getPaymentNotification($eventType);
            }

            $result[] = $notification;
        }

        return $result;
    }

    public static function invalidDataArrayDataProvider(): array
    {
        return [
            [[]],
            [['type' => 'test']],
            [['event' => 'test']],
            [['event' => new stdClass()]],
        ];
    }

    protected function getTestInstance(): NotificationFactory
    {
        return new NotificationFactory();
    }

    protected function getExpectedType(): string
    {
        return NotificationType::NOTIFICATION;
    }

    /**
     * @throws Exception
     */
    protected function getExpectedEvent(): mixed
    {
        return Random::value(NotificationEventType::getEnabledValues());
    }

    private function getRefundNotification(): array
    {
        $statuses = RefundStatus::getValidValues();
        $receiptRegistrations = ReceiptRegistrationStatus::getValidValues();

        $refund = [
            'id' => Random::str(36),
            'payment_id' => Random::str(36),
            'status' => Random::value($statuses),
            'amount' => [
                'value' => Random::float(0.01, 1000000.0),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
            'receipt_registration' => Random::value($receiptRegistrations),
            'description' => Random::str(1, 128),
        ];

        return [
            [
                'type' => $this->getExpectedType(),
                'event' => NotificationEventType::REFUND_SUCCEEDED,
                'object' => $refund,
            ],
        ];
    }

    private function getPaymentNotification($type): array
    {
        $statuses = PaymentStatus::getValidValues();
        $receiptRegistrations = ReceiptRegistrationStatus::getValidValues();

        $trueFalse = Random::bool();
        $confirmations = [
            [
                'type' => ConfirmationType::REDIRECT,
                'confirmation_url' => 'https://confirmation.url',
                'return_url' => 'https://merchant-site.ru/return_url',
                'enforce' => false,
            ],
            [
                'type' => ConfirmationType::EXTERNAL,
            ],
        ];

        $payment = [
            'id' => Random::str(36),
            'status' => Random::value($statuses),
            'recipient' => [
                'account_id' => Random::str(1, 64, '0123456789'),
                'gateway_id' => Random::str(1, 256),
            ],
            'amount' => [
                'value' => Random::float(0.01, 1000000.0),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'payment_method' => [
                'type' => PaymentMethodType::QIWI,
            ],
            'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
            'captured_at' => date(YOOKASSA_DATE, Random::int(1, time())),
            'confirmation' => Random::value($confirmations),
            'refunded' => [
                'value' => Random::float(0.01, 1000000.0),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'paid' => $trueFalse,
            'refundable' => $trueFalse,
            'receipt_registration' => Random::value($receiptRegistrations),
            'metadata' => [
                'value' => Random::str(1, 256),
                'currency' => Random::str(1, 256),
            ],
        ];

        return [
            [
                'type' => $this->getExpectedType(),
                'event' => $type,
                'object' => $payment,
            ],
        ];
    }

    private function getPayoutNotification($type): array
    {
        $cancellationDetailsParties = PayoutCancellationDetailsPartyCode::getValidValues();
        $cancellationDetailsReasons = PayoutCancellationDetailsReasonCode::getValidValues();
        $payoutDestinations = [
            PaymentMethodType::YOO_MONEY => [
                'type' => PaymentMethodType::YOO_MONEY,
                'account_number' => Random::str(11, 33, '1234567890'),
            ],
            PaymentMethodType::BANK_CARD => [
                'type' => PaymentMethodType::BANK_CARD,
                'card' => [
                    'first6' => Random::str(6, 6, '1234567890'),
                    'last4' => Random::str(4, 4, '1234567890'),
                    'card_type' => Random::value(BankCardType::getValidValues())
                ],
            ],
        ];

        $payout = [
            'id' => Random::str(36, 50),
            'status' => Random::value(PayoutStatus::getValidValues()),
            'amount' => ['value' => Random::int(1, 10000), 'currency' => 'RUB'],
            'description' => Random::str(1, Payout::MAX_LENGTH_DESCRIPTION),
            'payout_destination' => $payoutDestinations[Random::value([PaymentMethodType::YOO_MONEY, PaymentMethodType::BANK_CARD])],
            'created_at' => date(YOOKASSA_DATE, Random::int(111111111, time())),
            'test' => true,
            'deal' => ['id' => Random::str(36, 50)],
            'metadata' => ['order_id' => '37'],
            'cancellation_details' => [
                'party' => Random::value($cancellationDetailsParties),
                'reason' => Random::value($cancellationDetailsReasons),
            ],
        ];

        return [
            [
                'type' => $this->getExpectedType(),
                'event' => $type,
                'object' => $payout,
            ],
        ];
    }

    private function getDealNotification(): array
    {
        $statuses = DealStatus::getValidValues();
        $types = DealType::getValidValues();

        $trueFalse = Random::bool();
        $deal = [
            'id' => Random::str(36),
            'type' => Random::value($types),
            'status' => Random::value($statuses),
            'description' => Random::str(128),
            'balance' => [
                'value' => Random::float(0.01, 1000000.0),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'payout_balance' => [
                'value' => Random::float(0.01, 1000000.0),
                'currency' => Random::value(CurrencyCode::getValidValues()),
            ],
            'created_at' => date(YOOKASSA_DATE, Random::int(1, time())),
            'expires_at' => date(YOOKASSA_DATE, Random::int(1, time())),
            'fee_moment' => Random::value(FeeMoment::getEnabledValues()),
            'test' => $trueFalse,
            'metadata' => [],
        ];

        return [
            [
                'type' => $this->getExpectedType(),
                'event' => NotificationEventType::DEAL_CLOSED,
                'object' => $deal,
            ],
        ];
    }

    /**
     * @param mixed $event
     * @param mixed $object
     * @param mixed $value
     *
     * @throws Exception
     */
    private function assertObject(mixed $event, mixed $object, mixed $value): void
    {
        self::assertNotNull($object);

        switch ($event) {
            case NotificationEventType::REFUND_SUCCEEDED:
                self::assertInstanceOf(RefundResponse::class, $object);
                self::assertEquals($object, new RefundResponse($value));

                break;

            case NotificationEventType::PAYMENT_SUCCEEDED:
            case NotificationEventType::PAYMENT_WAITING_FOR_CAPTURE:
            case NotificationEventType::PAYMENT_CANCELED:
                self::assertInstanceOf(PaymentResponse::class, $object);
                self::assertEquals($object, new PaymentResponse($value));

                break;

            case NotificationEventType::PAYOUT_SUCCEEDED:
            case NotificationEventType::PAYOUT_CANCELED:
                self::assertInstanceOf(PayoutResponse::class, $object);
                self::assertEquals($object, new PayoutResponse($value));

                break;

            case NotificationEventType::DEAL_CLOSED:
                self::assertInstanceOf(DealResponse::class, $object);
                self::assertEquals($object, new DealResponse($value));

                break;
        }
    }
}
