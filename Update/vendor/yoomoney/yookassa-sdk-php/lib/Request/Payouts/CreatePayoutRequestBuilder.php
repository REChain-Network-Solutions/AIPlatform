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

namespace YooKassa\Request\Payouts;

use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payout\AbstractPayoutDestination;

/**
 * Класс, представляющий модель CreatePayoutRequestBuilder.
 *
 * Класс билдера объектов запросов к API на создание платежа.
 *
 * @example 02-builder.php 182 26 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CreatePayoutRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreatePayoutRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает сумму.
     *
     * @param AmountInterface|array|string $value Сумма выплаты
     *
     * @return self Инстанс билдера запросов
     */
    public function setAmount(mixed $value): self
    {
        $this->currentObject->setAmount($value);

        return $this;
    }

    /**
     * Устанавливает одноразовый токен для проведения выплаты.
     *
     * @param string|null $value Одноразовый токен для проведения выплаты
     *
     * @return CreatePayoutRequestBuilder Инстанс текущего билдера
     */
    public function setPayoutToken(?string $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setPayoutToken($value);

        return $this;
    }

    /**
     * Устанавливает объект с информацией для создания метода оплаты.
     *
     * @param null|AbstractPayoutDestination|array $value Объект создания метода оплаты или null
     */
    public function setPayoutDestinationData(mixed $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setPayoutDestinationData($value);

        return $this;
    }

    /**
     * Устанавливает идентификатор сохраненного способа оплаты.
     *
     * @param null|string $value Идентификатор сохраненного способа оплаты
     */
    public function setPaymentMethodId(?string $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setPaymentMethodId($value);

        return $this;
    }

    /**
     * Устанавливает сделку, в рамках которой нужно провести выплату.
     *
     * @param array|PayoutDealInfo $value Сделка, в рамках которой нужно провести выплату
     */
    public function setDeal(mixed $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setDeal($value);

        return $this;
    }

    /**
     * Устанавливает данные самозанятого, который получит выплату.
     *
     * @param null|array|PayoutSelfEmployedInfo $value Данные самозанятого, который получит выплату
     */
    public function setSelfEmployed(mixed $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setSelfEmployed($value);

        return $this;
    }

    /**
     * Устанавливает данные для формирования чека в сервисе Мой налог.
     *
     * @param null|array|IncomeReceiptData $value Данные для формирования чека в сервисе Мой налог
     */
    public function setReceiptData(mixed $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setReceiptData($value);

        return $this;
    }

    /**
     * Устанавливает персональные данные получателя выплаты.
     *
     * @param null|array|PayoutPersonalData[] $value Персональные данные получателя выплаты
     */
    public function setPersonalData(?array $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setPersonalData($value);

        return $this;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $value Метаданные платежа, устанавливаемые мерчантом
     *
     * @return CreatePayoutRequestBuilder Инстанс текущего билдера
     */
    public function setMetadata(mixed $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setMetadata($value);

        return $this;
    }

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $value Описание транзакции
     *
     * @return CreatePayoutRequestBuilder Инстанс текущего билдера
     */
    public function setDescription(?string $value): CreatePayoutRequestBuilder
    {
        $this->currentObject->setDescription($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return CreatePayoutRequestInterface|AbstractRequestInterface Инстанс объекта запроса
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreatePayoutRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): CreatePayoutRequest
    {
        return new CreatePayoutRequest();
    }
}
