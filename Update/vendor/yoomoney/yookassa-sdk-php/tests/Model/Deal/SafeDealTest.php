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

namespace Tests\YooKassa\Model\Deal;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Model\Metadata;

/**
 * SafeDealTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SafeDealTest extends AbstractTestCase
{
    protected SafeDeal $object;

    /**
     * @return SafeDeal
     */
    protected function getTestInstance(): SafeDeal
    {
        return new SafeDeal();
    }

    /**
     * @return void
     */
    public function testSafeDealClassExists(): void
    {
        $this->object = $this->getMockBuilder(SafeDeal::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SafeDeal::class));
        $this->assertInstanceOf(SafeDeal::class, $this->object);
    }

    /**
     * Test property "id"
     * @dataProvider validIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setId($value);
        self::assertNotNull($instance->getId());
        self::assertNotNull($instance->id);
        self::assertEquals($value, is_array($value) ? $instance->getId()->toArray() : $instance->getId());
        self::assertEquals($value, is_array($value) ? $instance->id->toArray() : $instance->id);
        self::assertLessThanOrEqual(50, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertLessThanOrEqual(50, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
        self::assertGreaterThanOrEqual(36, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertGreaterThanOrEqual(36, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
    }

    /**
     * Test invalid property "id"
     * @dataProvider invalidIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * Test property "type"
     * @dataProvider validTypeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setType($value);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertEquals($value, is_array($value) ? $instance->getType()->toArray() : $instance->getType());
        self::assertEquals($value, is_array($value) ? $instance->type->toArray() : $instance->type);
        self::assertContains($instance->getType(), ['safe_deal']);
        self::assertContains($instance->type, ['safe_deal']);
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
    public function validTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
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
     * Test property "fee_moment"
     * @dataProvider validFeeMomentDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFeeMoment(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setFeeMoment($value);
        self::assertNotNull($instance->getFeeMoment());
        self::assertNotNull($instance->fee_moment);
        self::assertEquals($value, is_array($value) ? $instance->getFeeMoment()->toArray() : $instance->getFeeMoment());
        self::assertEquals($value, is_array($value) ? $instance->fee_moment->toArray() : $instance->fee_moment);
    }

    /**
     * Test invalid property "fee_moment"
     * @dataProvider invalidFeeMomentDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFeeMoment(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFeeMoment($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFeeMomentDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_fee_moment'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFeeMomentDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_fee_moment'));
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
     * Test property "balance"
     * @dataProvider validBalanceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBalance(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBalance($value);
        self::assertNotNull($instance->getBalance());
        self::assertNotNull($instance->balance);
        self::assertEquals($value, is_array($value) ? $instance->getBalance()->toArray() : $instance->getBalance());
        self::assertEquals($value, is_array($value) ? $instance->balance->toArray() : $instance->balance);
    }

    /**
     * Test invalid property "balance"
     * @dataProvider invalidBalanceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBalance(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBalance($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBalanceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_balance'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBalanceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_balance'));
    }

    /**
     * Test property "payout_balance"
     * @dataProvider validPayoutBalanceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPayoutBalance(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPayoutBalance($value);
        self::assertNotNull($instance->getPayoutBalance());
        self::assertNotNull($instance->payout_balance);
        self::assertEquals($value, is_array($value) ? $instance->getPayoutBalance()->toArray() : $instance->getPayoutBalance());
        self::assertEquals($value, is_array($value) ? $instance->payout_balance->toArray() : $instance->payout_balance);
    }

    /**
     * Test invalid property "payout_balance"
     * @dataProvider invalidPayoutBalanceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPayoutBalance(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPayoutBalance($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPayoutBalanceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payout_balance'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPayoutBalanceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payout_balance'));
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
     * Test property "created_at"
     * @dataProvider validCreatedAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCreatedAt($value);
        self::assertNotNull($instance->getCreatedAt());
        self::assertNotNull($instance->created_at);
        if ($value instanceof Datetime) {
            self::assertEquals($value, $instance->getCreatedAt());
            self::assertEquals($value, $instance->created_at);
        } else {
            self::assertEquals(new Datetime($value), $instance->getCreatedAt());
            self::assertEquals(new Datetime($value), $instance->created_at);
        }
    }

    /**
     * Test invalid property "created_at"
     * @dataProvider invalidCreatedAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at'));
    }

    /**
     * Test property "expires_at"
     * @dataProvider validExpiresAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setExpiresAt($value);
        self::assertNotNull($instance->getExpiresAt());
        self::assertNotNull($instance->expires_at);
        if ($value instanceof Datetime) {
            self::assertEquals($value, $instance->getExpiresAt());
            self::assertEquals($value, $instance->expires_at);
        } else {
            self::assertEquals(new Datetime($value), $instance->getExpiresAt());
            self::assertEquals(new Datetime($value), $instance->expires_at);
        }
    }

    /**
     * Test invalid property "expires_at"
     * @dataProvider invalidExpiresAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
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
     * Test property "test"
     * @dataProvider validTestDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTest(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setTest($value);
        self::assertNotNull($instance->getTest());
        self::assertNotNull($instance->test);
        self::assertEquals($value, is_array($value) ? $instance->getTest()->toArray() : $instance->getTest());
        self::assertEquals($value, is_array($value) ? $instance->test->toArray() : $instance->test);
        self::assertIsBool($instance->getTest());
        self::assertIsBool($instance->test);
    }

    /**
     * Test invalid property "test"
     * @dataProvider invalidTestDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTest(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTest($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTestDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_test'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTestDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_test'));
    }
}
