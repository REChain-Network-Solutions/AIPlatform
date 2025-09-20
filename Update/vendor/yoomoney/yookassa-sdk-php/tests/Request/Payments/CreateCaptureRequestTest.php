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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Request\Payments\CreateCaptureRequest;
use YooKassa\Request\Payments\CreateCaptureRequestBuilder;

/**
 * CreateCaptureRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateCaptureRequestTest extends AbstractTestCase
{
    protected CreateCaptureRequest $object;

    /**
     * @param mixed|null $value
     * @return CreateCaptureRequest
     */
    protected function getTestInstance(mixed $value = null): CreateCaptureRequest
    {
        return new CreateCaptureRequest($value);
    }

    /**
     * @return void
     */
    public function testCreateCaptureRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(CreateCaptureRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CreateCaptureRequest::class));
        $this->assertInstanceOf(CreateCaptureRequest::class, $this->object);
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
        self::assertEmpty($instance->getAmount());
        self::assertEmpty($instance->amount);
        $instance->setAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getAmount()->toArray() : $instance->getAmount());
        self::assertEquals($value, is_array($value) ? $instance->amount->toArray() : $instance->amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getAmount());
            self::assertNotNull($instance->amount);
        }
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
     * Test valid method "validate"
     *
     * @return void
     */
    public function testValidate(): void
    {
        $instance = new CreateCaptureRequest();
        self::assertTrue($instance->validate());
        $amount = new MonetaryAmount();
        $instance->setAmount($amount);
        self::assertFalse($instance->validate());
        $amount->setValue(1);
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

        self::assertNotNull($instance->getReceipt());
        $instance->removeReceipt();
        self::assertTrue($instance->validate());
        self::assertNull($instance->getReceipt());

        $instance->setAmount(new MonetaryAmount());
        self::assertFalse($instance->validate());
    }

    /**
     * Test valid method "builder"
     *
     * @return void
     */
    public function testBuilder(): void
    {
        $builder = CreateCaptureRequest::builder();
        self::assertInstanceOf(CreateCaptureRequestBuilder::class, $builder);
    }
}
