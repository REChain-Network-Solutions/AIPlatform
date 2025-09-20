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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Model\Deal\CaptureDealData;

/**
 * Класс, представляющий модель CreateCaptureRequestBuilder.
 *
 * Класс билдера объекта запроса подтверждения платежа, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CreateCaptureRequestBuilder extends AbstractPaymentRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreateCaptureRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Осуществляет сборку объекта запроса к API.
     *
     * @param null|array $options Массив дополнительных настроек объекта
     *
     * @return CreateCaptureRequest|AbstractRequestInterface Инстанс объекта запроса к API
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }
        if ($this->amount->getValue() > 0) {
            $this->currentObject->setAmount($this->amount);
        }
        if ($this->receipt->notEmpty()) {
            $this->currentObject->setReceipt($this->receipt);
        }

        return parent::build();
    }

    /**
     * Устанавливает информацию об авиабилетах
     * @param AirlineInterface|array|null $value Объект данных длинной записи или ассоциативный массив с данными
     *
     * @return CreateCaptureRequestBuilder
     */
    public function setAirline(mixed $value): self
    {
        $this->currentObject->setAirline($value);

        return $this;
    }

    /**
     * Устанавливает сделку.
     *
     * @param null|array|CaptureDealData $value Данные о сделке, в составе подтверждения оплаты
     *
     * @return CreateCaptureRequestBuilder Инстанс билдера запросов
     */
    public function setDeal(mixed $value): CreateCaptureRequestBuilder
    {
        $this->currentObject->setDeal($value);

        return $this;
    }

    protected function initCurrentObject(): CreateCaptureRequest
    {
        parent::initCurrentObject();

        return new CreateCaptureRequest();
    }
}
