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
use YooKassa\Common\AbstractObject;
use YooKassa\Helpers\Random;
use YooKassa\Model\CurrencyCode;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\Transfer;
use YooKassa\Model\Receipt\PaymentMode;
use YooKassa\Model\Receipt\PaymentSubject;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Request\Payments\CreatePaymentRequest;
use YooKassa\Request\Payments\CreatePaymentRequestBuilder;
use YooKassa\Request\Payments\PaymentData\PaymentDataQiwi;

/**
 * CreatePaymentRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreatePaymentRequestTest extends AbstractTestCase
{
    protected CreatePaymentRequest $object;

    /**
     * @return CreatePaymentRequest
     */
    protected function getTestInstance(): CreatePaymentRequest
    {
        return new CreatePaymentRequest();
    }

    /**
     * @return void
     */
    public function testCreatePaymentRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(CreatePaymentRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CreatePaymentRequest::class));
        $this->assertInstanceOf(CreatePaymentRequest::class, $this->object);
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
            self::assertLessThanOrEqual(128, is_string($instance->getDescription()) ? mb_strlen($instance->getDescription()) : $instance->getDescription());
            self::assertLessThanOrEqual(128, is_string($instance->description) ? mb_strlen($instance->description) : $instance->description);
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
     * Test property "recipient"
     * @dataProvider validRecipientDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRecipient(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRecipient());
        self::assertEmpty($instance->recipient);
        $instance->setRecipient($value);
        self::assertEquals($value, is_array($value) ? $instance->getRecipient()->toArray() : $instance->getRecipient());
        self::assertEquals($value, is_array($value) ? $instance->recipient->toArray() : $instance->recipient);
        if (!empty($value)) {
            self::assertTrue($instance->hasRecipient());
            self::assertNotNull($instance->getRecipient());
            self::assertNotNull($instance->recipient);
        }
    }

    /**
     * Test invalid property "recipient"
     * @dataProvider invalidRecipientDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRecipient(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRecipient($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRecipientDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRecipientDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient'));
    }

    /**
     * Test property "payment_token"
     * @dataProvider validPaymentTokenDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentToken(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentToken());
        self::assertEmpty($instance->payment_token);
        $instance->setPaymentToken($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentToken()->toArray() : $instance->getPaymentToken());
        self::assertEquals($value, is_array($value) ? $instance->payment_token->toArray() : $instance->payment_token);
        if (!empty($value)) {
            self::assertTrue($instance->hasPaymentToken());
            self::assertNotNull($instance->getPaymentToken());
            self::assertNotNull($instance->payment_token);
        }
    }

    /**
     * Test invalid property "payment_token"
     * @dataProvider invalidPaymentTokenDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentToken(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentToken($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_token'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_token'));
    }

    /**
     * Test property "payment_method_id"
     * @dataProvider validPaymentMethodIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentMethodId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentMethodId());
        self::assertEmpty($instance->payment_method_id);
        $instance->setPaymentMethodId($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentMethodId()->toArray() : $instance->getPaymentMethodId());
        self::assertEquals($value, is_array($value) ? $instance->payment_method_id->toArray() : $instance->payment_method_id);
        if (!empty($value)) {
            self::assertTrue($instance->hasPaymentMethodId());
            self::assertNotNull($instance->getPaymentMethodId());
            self::assertNotNull($instance->payment_method_id);
        }
    }

    /**
     * Test invalid property "payment_method_id"
     * @dataProvider invalidPaymentMethodIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentMethodId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentMethodId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentMethodIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentMethodIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_id'));
    }

    /**
     * Test property "payment_method_data"
     * @dataProvider validPaymentMethodDataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentMethodData(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentMethodData());
        self::assertEmpty($instance->payment_method_data);
        $instance->setPaymentMethodData($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentMethodData()->toArray() : $instance->getPaymentMethodData());
        self::assertEquals($value, is_array($value) ? $instance->payment_method_data->toArray() : $instance->payment_method_data);
        if (!empty($value)) {
            self::assertTrue($instance->hasPaymentMethodData());
            self::assertNotNull($instance->getPaymentMethodData());
            self::assertNotNull($instance->payment_method_data);
        }
    }

    /**
     * Test invalid property "payment_method_data"
     * @dataProvider invalidPaymentMethodDataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentMethodData(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentMethodData($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentMethodDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_data'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentMethodDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_data'));
    }

    /**
     * Test property "confirmation"
     * @dataProvider validConfirmationDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testConfirmation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getConfirmation());
        self::assertEmpty($instance->confirmation);
        $instance->setConfirmation($value);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmation()->toArray() : $instance->getConfirmation());
        self::assertEquals($value, is_array($value) ? $instance->confirmation->toArray() : $instance->confirmation);
        if (!empty($value)) {
            self::assertTrue($instance->hasConfirmation());
            self::assertNotNull($instance->getConfirmation());
            self::assertNotNull($instance->confirmation);
        }
    }

    /**
     * Test invalid property "confirmation"
     * @dataProvider invalidConfirmationDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidConfirmation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmation($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
     * Test property "save_payment_method"
     * @dataProvider validSavePaymentMethodDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSavePaymentMethod(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSavePaymentMethod());
        self::assertEmpty($instance->save_payment_method);
        $instance->setSavePaymentMethod($value);
        self::assertEquals($value, is_array($value) ? $instance->getSavePaymentMethod()->toArray() : $instance->getSavePaymentMethod());
        self::assertEquals($value, is_array($value) ? $instance->save_payment_method->toArray() : $instance->save_payment_method);
        if (!empty($value)) {
            self::assertTrue($instance->hasSavePaymentMethod());
            self::assertNotNull($instance->getSavePaymentMethod());
            self::assertNotNull($instance->save_payment_method);
            self::assertIsBool($instance->getSavePaymentMethod());
            self::assertIsBool($instance->save_payment_method);
            self::assertIsBool($instance->getSavePaymentMethod());
            self::assertIsBool($instance->save_payment_method);
        }
    }

    /**
     * Test invalid property "save_payment_method"
     * @dataProvider invalidSavePaymentMethodDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSavePaymentMethod(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSavePaymentMethod($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSavePaymentMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_save_payment_method'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSavePaymentMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_save_payment_method'));
    }

    /**
     * Test property "capture"
     * @dataProvider validCaptureDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapture(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCapture($value);
        self::assertEquals($value, is_array($value) ? $instance->getCapture()->toArray() : $instance->getCapture());
        self::assertEquals($value, is_array($value) ? $instance->capture->toArray() : $instance->capture);
        if (!empty($value)) {
            self::assertTrue($instance->hasCapture());
            self::assertNotNull($instance->getCapture());
            self::assertNotNull($instance->capture);
            self::assertIsBool($instance->getCapture());
            self::assertIsBool($instance->capture);
            self::assertIsBool($instance->getCapture());
            self::assertIsBool($instance->capture);
        }
    }

    /**
     * Test invalid property "capture"
     * @dataProvider invalidCaptureDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapture(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapture($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCaptureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_capture'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCaptureDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_capture'));
    }

    /**
     * Test property "client_ip"
     * @dataProvider validClientIpDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testClientIp(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getClientIp());
        self::assertEmpty($instance->client_ip);
        $instance->setClientIp($value);
        self::assertEquals($value, is_array($value) ? $instance->getClientIp()->toArray() : $instance->getClientIp());
        self::assertEquals($value, is_array($value) ? $instance->client_ip->toArray() : $instance->client_ip);
        if (!empty($value)) {
            self::assertTrue($instance->hasClientIp());
            self::assertNotNull($instance->getClientIp());
            self::assertNotNull($instance->client_ip);
        }
    }

    /**
     * Test invalid property "client_ip"
     * @dataProvider invalidClientIpDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidClientIp(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setClientIp($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validClientIpDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_client_ip'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidClientIpDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_client_ip'));
    }

    /**
     * Test property "metadata"
     * @dataProvider validMetadataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMetadata(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMetadata());
        self::assertEmpty($instance->metadata);
        $instance->setMetadata($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasMetadata());
            self::assertNotNull($instance->getMetadata());
            self::assertNotNull($instance->metadata);
            foreach ($value as $key => $element) {
                if (!empty($element)) {
                    self::assertEquals($element, $instance->getMetadata()[$key]);
                    self::assertEquals($element, $instance->metadata[$key]);
                    self::assertIsObject($instance->getMetadata());
                    self::assertIsObject($instance->metadata);
                }
            }
            self::assertCount(count($value), $instance->getMetadata());
            self::assertCount(count($value), $instance->metadata);
            if ($instance->getMetadata() instanceof Metadata) {
                self::assertEquals($value, $instance->getMetadata()->toArray());
                self::assertEquals($value, $instance->metadata->toArray());
                self::assertCount(count($value), $instance->getMetadata());
                self::assertCount(count($value), $instance->metadata);
            }
        }
    }

    /**
     * Test invalid property "metadata"
     * @dataProvider invalidMetadataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMetadata(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMetadata($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * Test property "airline"
     * @dataProvider validAirlineDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAirline(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAirline());
        self::assertEmpty($instance->airline);
        $instance->setAirline($value);
        self::assertEquals($value, is_array($value) ? $instance->getAirline()->toArray() : $instance->getAirline());
        self::assertEquals($value, is_array($value) ? $instance->airline->toArray() : $instance->airline);
        if (!empty($value)) {
            self::assertTrue($instance->hasAirline());
            self::assertNotNull($instance->getAirline());
            self::assertNotNull($instance->airline);
        }
    }

    /**
     * Test invalid property "airline"
     * @dataProvider invalidAirlineDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAirline(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAirline($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAirlineDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_airline'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAirlineDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_airline'));
    }

    /**
     * Test property "transfers"
     * @dataProvider validTransfersDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTransfers(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getTransfers());
        self::assertEmpty($instance->transfers);
        self::assertIsObject($instance->getTransfers());
        self::assertIsObject($instance->transfers);
        self::assertCount(0, $instance->getTransfers());
        self::assertCount(0, $instance->transfers);
        $instance->setTransfers($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasTransfers());
            self::assertNotNull($instance->getTransfers());
            self::assertNotNull($instance->transfers);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getTransfers()[$key]->toArray());
                    self::assertEquals($element, $instance->transfers[$key]->toArray());
                    self::assertIsArray($instance->getTransfers()[$key]->toArray());
                    self::assertIsArray($instance->transfers[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getTransfers()->get($key));
                    self::assertIsObject($instance->getTransfers()->get($key));
                    self::assertIsObject($instance->transfers->get($key));
                    self::assertIsObject($instance->getTransfers());
                    self::assertIsObject($instance->transfers);
                }
            }
            self::assertCount(count($value), $instance->getTransfers());
            self::assertCount(count($value), $instance->transfers);
        }
    }

    /**
     * Test invalid property "transfers"
     * @dataProvider invalidTransfersDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTransfers(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTransfers($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTransfersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_transfers'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTransfersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_transfers'));
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
     * Test property "merchant_customer_id"
     * @dataProvider validMerchantCustomerIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMerchantCustomerId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMerchantCustomerId());
        self::assertEmpty($instance->merchant_customer_id);
        $instance->setMerchantCustomerId($value);
        self::assertEquals($value, is_array($value) ? $instance->getMerchantCustomerId()->toArray() : $instance->getMerchantCustomerId());
        self::assertEquals($value, is_array($value) ? $instance->merchant_customer_id->toArray() : $instance->merchant_customer_id);
        if (!empty($value)) {
            self::assertTrue($instance->hasMerchantCustomerId());
            self::assertNotNull($instance->getMerchantCustomerId());
            self::assertNotNull($instance->merchant_customer_id);
        }
    }

    /**
     * Test invalid property "merchant_customer_id"
     * @dataProvider invalidMerchantCustomerIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMerchantCustomerId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMerchantCustomerId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMerchantCustomerIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMerchantCustomerIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_id'));
    }

    /**
     * Test property "payment_order"
     * @dataProvider validPaymentOrderDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentOrder(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentOrder());
        self::assertEmpty($instance->payment_order);
        $instance->setPaymentOrder($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentOrder()->toArray() : $instance->getPaymentOrder());
        self::assertEquals($value, is_array($value) ? $instance->payment_order->toArray() : $instance->payment_order);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentOrder());
            self::assertNotNull($instance->hasPaymentOrder());
            self::assertNotNull($instance->payment_order);
        }
    }

    /**
     * Test invalid property "payment_order"
     * @dataProvider invalidPaymentOrderDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentOrder(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentOrder($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentOrderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_order'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentOrderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_order'));
    }

    /**
     * Test property "receiver"
     * @dataProvider validReceiverDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReceiver(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReceiver());
        self::assertEmpty($instance->receiver);
        $instance->setReceiver($value);
        self::assertEquals($value, is_array($value) ? $instance->getReceiver()->toArray() : $instance->getReceiver());
        self::assertEquals($value, is_array($value) ? $instance->receiver->toArray() : $instance->receiver);
        if (!empty($value)) {
            self::assertTrue($instance->hasReceiver());
            self::assertNotNull($instance->getReceiver());
            self::assertNotNull($instance->receiver);
        }
    }

    /**
     * Test invalid property "receiver"
     * @dataProvider invalidReceiverDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReceiver(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReceiver($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReceiverDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_receiver'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReceiverDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_receiver'));
    }

    public function testValidate(): void
    {
        $instance = new CreatePaymentRequest();

        self::assertFalse($instance->validate());

        $amount = new MonetaryAmount();
        $instance->setAmount($amount);
        self::assertFalse($instance->validate());

        $instance->setAmount(new MonetaryAmount(10));
        self::assertTrue($instance->validate());

        $instance->setPaymentToken(Random::str(10));
        self::assertTrue($instance->validate());
        $instance->setPaymentMethodId(Random::str(10));
        self::assertFalse($instance->validate());
        $instance->setPaymentMethodId(null);
        self::assertTrue($instance->validate());
        $instance->setPaymentMethodData(new PaymentDataQiwi(['phone' => Random::str(11, '0123456789')]));
        self::assertFalse($instance->validate());
        $instance->setPaymentToken(null);
        self::assertTrue($instance->validate());
        $instance->setPaymentMethodId(Random::str(10));
        self::assertFalse($instance->validate());
        $instance->setPaymentMethodId(null);
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

        $transfers = $this->validClassDataProvider(new Transfer());
        $instance->setTransfers($transfers);
        self::assertFalse($instance->validate());
    }

    /**
     * @param AbstractObject $class
     * @return array
     * @throws \ReflectionException
     * @throws \ReverseRegex\Exception
     */
    public function validClassDataProvider(AbstractObject $class): array
    {
        return $this->getValidDataProviderByClass($class);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTransferDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_transfers'));
    }

    /**
     * Test valid method "builder"
     *
     * @return void
     */
    public function testBuilder(): void
    {
        $builder = CreatePaymentRequest::builder();
        self::assertInstanceOf(CreatePaymentRequestBuilder::class, $builder);
    }
}
