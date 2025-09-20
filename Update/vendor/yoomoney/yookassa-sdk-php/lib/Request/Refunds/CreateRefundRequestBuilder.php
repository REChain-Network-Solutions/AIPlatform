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

namespace YooKassa\Request\Refunds;

use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Deal\RefundDealData;
use YooKassa\Model\Refund\SourceInterface;
use YooKassa\Request\Payments\AbstractPaymentRequestBuilder;
use YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData;

/**
 * Класс, представляющий модель CreateRefundRequestBuilder.
 *
 * Класс билдера запросов к API на создание возврата средств.
 *
 * @example 02-builder.php 147 33 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CreateRefundRequestBuilder extends AbstractPaymentRequestBuilder
{
    /**
     * Собираемый объект запроса к API.
     *
     * @var CreateRefundRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает айди платежа для которого создаётся возврат
     *
     * @param string|null $value Айди платежа
     *
     * @return CreateRefundRequestBuilder Инстанс текущего билдера
     *
     * @throws EmptyPropertyValueException Выбрасывается если передано пустое значение айди платежа
     * @throws InvalidPropertyValueException Выбрасывается если переданное значение является строкой, но не является
     * валидным значением айди платежа
     * @throws InvalidPropertyValueTypeException Выбрасывается если передано значение не валидного типа
     */
    public function setPaymentId(?string $value): CreateRefundRequestBuilder
    {
        $this->currentObject->setPaymentId($value);

        return $this;
    }

    /**
     * Устанавливает комментарий к возврату.
     *
     * @param string|null $value Комментарий к возврату
     *
     * @return CreateRefundRequestBuilder Инстанс текущего билдера
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если была передана не строка
     */
    public function setDescription(?string $value): CreateRefundRequestBuilder
    {
        $this->currentObject->setDescription($value);

        return $this;
    }

    /**
     * Устанавливает источники возврата.
     *
     * @param array|SourceInterface[]|ListObjectInterface $value Массив трансферов
     *
     * @return self Инстанс билдера запросов
     */
    public function setSources(array $value): self
    {
        $this->currentObject->setSources($value);

        return $this;
    }

    /**
     * Добавляет источник возврата.
     *
     * @param array|SourceInterface $value Источник возврата
     *
     * @return self Инстанс билдера запросов
     */
    public function addSource(mixed $value): self
    {
        $this->currentObject->getSources()->add($value);

        return $this;
    }

    /**
     * Устанавливает сделку.
     *
     * @param null|array|RefundDealData $value Данные о сделке, в составе которой проходит возврат
     *
     * @return CreateRefundRequestBuilder Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException
     */
    public function setDeal(mixed $value): CreateRefundRequestBuilder
    {
        $this->currentObject->setDeal($value);

        return $this;
    }

    /**
     * Устанавливает метод возврата.
     *
     * @param AbstractRefundMethodData|array|null $value
     *
     * @return self
     */
    public function setRefundMethodData(mixed $value = null): self
    {
        $this->currentObject->setRefundMethodData($value);

        return $this;
    }

    /**
     * Строит объект запроса к API.
     *
     * @param null|array $options Устанавливаемые параметры запроса
     *
     * @return CreateRefundRequest|AbstractRequestInterface Инстанс сгенерированного объекта запроса к API
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }

        $this->currentObject->setAmount($this->amount);

        if ($this->receipt->notEmpty()) {
            $this->currentObject->setReceipt($this->receipt);
        }

        return parent::build();
    }

    /**
     * Возвращает новый объект для сборки.
     *
     * @return CreateRefundRequest Собираемый объект запроса к API
     */
    protected function initCurrentObject(): CreateRefundRequest
    {
        parent::initCurrentObject();

        return new CreateRefundRequest();
    }
}
