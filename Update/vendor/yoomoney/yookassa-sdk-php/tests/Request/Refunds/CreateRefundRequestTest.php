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

namespace Tests\YooKassa\Request\Refunds;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Deal\SettlementPayoutPaymentType;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Request\Refunds\CreateRefundRequest;

/**
 * CreateRefundRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateRefundRequestTest extends AbstractTestCase
{
    protected CreateRefundRequest $object;

    /**
     * @param mixed|null $value
     * @return CreateRefundRequest
     */
    protected function getTestInstance(mixed $value = null): CreateRefundRequest
    {
        return new CreateRefundRequest($value);
    }

    /**
     * @return void
     */
    public function testCreateRefundRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(CreateRefundRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CreateRefundRequest::class));
        $this->assertInstanceOf(CreateRefundRequest::class, $this->object);
    }

    /**
     * Test property "payment_id"
     * @dataProvider validPaymentIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPaymentId($value);
        self::assertNotNull($instance->getPaymentId());
        self::assertNotNull($instance->payment_id);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentId()->toArray() : $instance->getPaymentId());
        self::assertEquals($value, is_array($value) ? $instance->payment_id->toArray() : $instance->payment_id);
        self::assertLessThanOrEqual(36, is_string($instance->getPaymentId()) ? mb_strlen($instance->getPaymentId()) : $instance->getPaymentId());
        self::assertLessThanOrEqual(36, is_string($instance->payment_id) ? mb_strlen($instance->payment_id) : $instance->payment_id);
        self::assertGreaterThanOrEqual(36, is_string($instance->getPaymentId()) ? mb_strlen($instance->getPaymentId()) : $instance->getPaymentId());
        self::assertGreaterThanOrEqual(36, is_string($instance->payment_id) ? mb_strlen($instance->payment_id) : $instance->payment_id);
    }

    /**
     * Test invalid property "payment_id"
     * @dataProvider invalidPaymentIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_id'));
    }

    /**
     * Test property "amount"
     * @dataProvider validAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAmount($value);
        self::assertNotNull($instance->getAmount());
        self::assertNotNull($instance->amount);
        self::assertEquals($value, is_array($value) ? $instance->getAmount()->toArray() : $instance->getAmount());
        self::assertEquals($value, is_array($value) ? $instance->amount->toArray() : $instance->amount);
    }

    /**
     * Test invalid property "amount"
     * @dataProvider invalidAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }

    /**
     * Test property "description"
     * @dataProvider validDescriptionDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDescription(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDescription());
        self::assertEmpty($instance->description);
        $instance->setDescription($value);
        self::assertEquals($value, is_array($value) ? $instance->getDescription()->toArray() : $instance->getDescription());
        self::assertEquals($value, is_array($value) ? $instance->description->toArray() : $instance->description);
        if (!empty($value)) {
            self::assertTrue($instance->hasDescription());
            self::assertNotNull($instance->getDescription());
            self::assertNotNull($instance->description);
            self::assertLessThanOrEqual(250, is_string($instance->getDescription()) ? mb_strlen($instance->getDescription()) : $instance->getDescription());
            self::assertLessThanOrEqual(250, is_string($instance->description) ? mb_strlen($instance->description) : $instance->description);
        }
    }

    /**
     * Test invalid property "description"
     * @dataProvider invalidDescriptionDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDescription(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDescription($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
    }

    /**
     * Test property "receipt"
     * @dataProvider validReceiptDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReceipt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReceipt());
        self::assertEmpty($instance->receipt);
        $instance->setReceipt($value);
        self::assertEquals($value, is_array($value) ? $instance->getReceipt()->toArray() : $instance->getReceipt());
        self::assertEquals($value, is_array($value) ? $instance->receipt->toArray() : $instance->receipt);
        if (!empty($value)) {
            self::assertNotNull($instance->getReceipt());
            self::assertNotNull($instance->receipt);
        }
    }

    /**
     * Test invalid property "receipt"
     * @dataProvider invalidReceiptDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReceipt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReceipt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReceiptDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReceiptDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt'));
    }

    /**
     * Test property "sources"
     * @dataProvider validSourcesDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSources(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSources());
        self::assertEmpty($instance->sources);
        self::assertIsObject($instance->getSources());
        self::assertIsObject($instance->sources);
        self::assertCount(0, $instance->getSources());
        self::assertCount(0, $instance->sources);
        $instance->setSources($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasSources());
            self::assertNotNull($instance->getSources());
            self::assertNotNull($instance->sources);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSources()[$key]->toArray());
                    self::assertEquals($element, $instance->sources[$key]->toArray());
                    self::assertIsArray($instance->getSources()[$key]->toArray());
                    self::assertIsArray($instance->sources[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSources()->get($key));
                    self::assertIsObject($instance->getSources()->get($key));
                    self::assertIsObject($instance->sources->get($key));
                    self::assertIsObject($instance->getSources());
                    self::assertIsObject($instance->sources);
                }
            }
            self::assertCount(count($value), $instance->getSources());
            self::assertCount(count($value), $instance->sources);
        }
    }

    /**
     * Test invalid property "sources"
     * @dataProvider invalidSourcesDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSources(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSources($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSourcesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_sources'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSourcesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_sources'));
    }

    /**
     * Test property "deal"
     * @dataProvider validDealDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDeal(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDeal());
        self::assertEmpty($instance->deal);
        $instance->setDeal($value);
        self::assertEquals($value, is_array($value) ? $instance->getDeal()->toArray() : $instance->getDeal());
        self::assertEquals($value, is_array($value) ? $instance->deal->toArray() : $instance->deal);
        if (!empty($value)) {
            self::assertTrue($instance->hasDeal());
            self::assertNotNull($instance->getDeal());
            self::assertNotNull($instance->deal);
        }
    }

    /**
     * Test invalid property "deal"
     * @dataProvider invalidDealDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDeal(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDeal($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDealDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_deal'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDealDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_deal'));
    }

    /**
     * Test property "refund_method_data"
     * @dataProvider validRefundMethodDataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundMethodData(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRefundMethodData());
        self::assertEmpty($instance->refund_method_data);
        $instance->setRefundMethodData($value);
        self::assertEquals($value, is_array($value) ? $instance->getRefundMethodData()->toArray() : $instance->getRefundMethodData());
        self::assertEquals($value, is_array($value) ? $instance->refund_method_data->toArray() : $instance->refund_method_data);
        if (!empty($value)) {
            self::assertTrue($instance->hasRefundMethodData());
            self::assertNotNull($instance->getRefundMethodData());
            self::assertNotNull($instance->refund_method_data);
        }
    }

    /**
     * Test invalid property "refund_method_data"
     * @dataProvider invalidRefundMethodDataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundMethodData(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundMethodData($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundMethodDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_method_data'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundMethodDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_method_data'));
    }

    public function testValidate(): void
    {
        $instance = new CreateRefundRequest();

        self::assertFalse($instance->validate());
        $instance->setPaymentId(Random::str(36));
        self::assertFalse($instance->validate());
        $instance->setAmount(new MonetaryAmount());
        self::assertFalse($instance->validate());
        $instance->setAmount(new MonetaryAmount(Random::int(1, 100000)));
        self::assertTrue($instance->validate());
        $instance = new CreateRefundRequest();
        $instance->setDeal([
            'refund_settlements' => [
                [
                    'type' => Random::value(SettlementPayoutPaymentType::getValidValues()),
                    'amount' => [
                        'value' => round(Random::float(1.00, 100.00), 2),
                        'currency' => 'RUB',
                    ],
                ],
            ],
        ]);
        self::assertFalse($instance->validate());
        $instance->setPaymentId(Random::str(36));
        $instance->setAmount(new MonetaryAmount(Random::int(1, 100000)));
        self::assertTrue($instance->validate());

        $receipt = new Receipt();
        $receipt->setItems([
            [
                'description' => Random::str(10),
                'quantity' => (float) Random::int(1, 10),
                'amount' => [
                    'value' => round(Random::float(1, 100), 2),
                    'currency' => CurrencyCode::RUB,
                ],
                'vat_code' => Random::int(1, 6),
                'payment_subject' => PaymentSubject::COMMODITY,
                'payment_mode' => PaymentMode::PARTIAL_PREPAYMENT,
            ],
        ]);
        $instance->setReceipt($receipt);
        $item = new ReceiptItem();
        $item->setPrice(new ReceiptItemAmount(10));
        $item->setDescription('test');
        $receipt->addItem($item);
        self::assertFalse($instance->validate());
        $receipt->getCustomer()->setPhone('123123');
        self::assertTrue($instance->validate());
        $item->setVatCode(3);
        self::assertTrue($instance->validate());
        $receipt->setTaxSystemCode(4);
        self::assertTrue($instance->validate());

        self::assertTrue($instance->hasReceipt());
        $instance->removeReceipt();
        self::assertFalse($instance->hasReceipt());
    }

    /**
     * Test valid method "builder"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testBuilder(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertInstanceOf(AbstractRequestBuilder::class, $instance::builder());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }
}
