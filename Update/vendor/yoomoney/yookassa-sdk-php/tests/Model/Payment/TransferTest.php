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

namespace Tests\YooKassa\Model\Payment;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Transfer;

/**
 * TransferTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class TransferTest extends AbstractTestCase
{
    protected Transfer $object;

    /**
     * @return Transfer
     */
    protected function getTestInstance(): Transfer
    {
        return new Transfer();
    }

    /**
     * @return void
     */
    public function testTransferClassExists(): void
    {
        $this->object = $this->getMockBuilder(Transfer::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Transfer::class));
        $this->assertInstanceOf(Transfer::class, $this->object);
    }

    /**
     * Test property "account_id"
     * @dataProvider validAccountIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAccountId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAccountId($value);
        self::assertNotNull($instance->getAccountId());
        self::assertNotNull($instance->account_id);
        self::assertEquals($value, is_array($value) ? $instance->getAccountId()->toArray() : $instance->getAccountId());
        self::assertEquals($value, is_array($value) ? $instance->account_id->toArray() : $instance->account_id);
    }

    /**
     * Test invalid property "account_id"
     * @dataProvider invalidAccountIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAccountId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAccountId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAccountIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAccountIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_id'));
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
     * Test property "status"
     * @dataProvider validStatusDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testStatus(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setStatus($value);
        self::assertNotNull($instance->getStatus());
        self::assertNotNull($instance->status);
        self::assertEquals($value, is_array($value) ? $instance->getStatus()->toArray() : $instance->getStatus());
        self::assertEquals($value, is_array($value) ? $instance->status->toArray() : $instance->status);
    }

    /**
     * Test invalid property "status"
     * @dataProvider invalidStatusDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidStatus(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setStatus($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * Test property "platform_fee_amount"
     * @dataProvider validPlatformFeeAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPlatformFeeAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPlatformFeeAmount());
        self::assertEmpty($instance->platform_fee_amount);
        $instance->setPlatformFeeAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getPlatformFeeAmount()->toArray() : $instance->getPlatformFeeAmount());
        self::assertEquals($value, is_array($value) ? $instance->platform_fee_amount->toArray() : $instance->platform_fee_amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getPlatformFeeAmount());
            self::assertNotNull($instance->platform_fee_amount);
        }
    }

    /**
     * Test invalid property "platform_fee_amount"
     * @dataProvider invalidPlatformFeeAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPlatformFeeAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPlatformFeeAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPlatformFeeAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_platform_fee_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPlatformFeeAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_platform_fee_amount'));
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
     * Test property "release_funds"
     * @dataProvider validReleaseFundsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReleaseFunds(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReleaseFunds());
        self::assertEmpty($instance->release_funds);
        $instance->setReleaseFunds($value);
        self::assertEquals($value, is_array($value) ? $instance->getReleaseFunds()->toArray() : $instance->getReleaseFunds());
        self::assertEquals($value, is_array($value) ? $instance->release_funds->toArray() : $instance->release_funds);
        if (!empty($value)) {
            self::assertNotNull($instance->getReleaseFunds());
            self::assertNotNull($instance->release_funds);
            self::assertIsBool($instance->getReleaseFunds());
            self::assertIsBool($instance->release_funds);
            self::assertIsBool($instance->getReleaseFunds());
            self::assertIsBool($instance->release_funds);
        }
    }

    /**
     * Test invalid property "release_funds"
     * @dataProvider invalidReleaseFundsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReleaseFunds(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReleaseFunds($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReleaseFundsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_release_funds'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReleaseFundsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_release_funds'));
    }

    /**
     * Test property "connected_account_id"
     * @dataProvider validConnectedAccountIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testConnectedAccountId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getConnectedAccountId());
        self::assertEmpty($instance->connected_account_id);
        $instance->setConnectedAccountId($value);
        self::assertEquals($value, is_array($value) ? $instance->getConnectedAccountId()->toArray() : $instance->getConnectedAccountId());
        self::assertEquals($value, is_array($value) ? $instance->connected_account_id->toArray() : $instance->connected_account_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getConnectedAccountId());
            self::assertNotNull($instance->connected_account_id);
        }
    }

    /**
     * Test invalid property "connected_account_id"
     * @dataProvider invalidConnectedAccountIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidConnectedAccountId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConnectedAccountId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validConnectedAccountIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_connected_account_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidConnectedAccountIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_connected_account_id'));
    }
}
