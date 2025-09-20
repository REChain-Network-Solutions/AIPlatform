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

namespace YooKassa\Request\PaymentMethods;

use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation;

/**
 * Класс, представляющий модель CreatePaymentMethodRequestBuilder.
 *
 * Класс билдера объекта запроса на создание платежа, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 02-builder.php 272 23 Пример использования билдера
 */
class CreatePaymentMethodRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreatePaymentMethodRequestInterface|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает type.
     *
     * @param string|null $value Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @return self
     */
    public function setType(?string $value = null): CreatePaymentMethodRequestBuilder
    {
        $this->currentObject->setType($value);

        return $this;
    }

    /**
     * Устанавливает card.
     *
     * @param PaymentMethodCard|array|null $value Данные для проверки и сохранения банковской карты.
     *
     * @return self
     */
    public function setCard(mixed $value = null): CreatePaymentMethodRequestBuilder
    {
        $this->currentObject->setCard($value);

        return $this;
    }

    /**
     * Устанавливает holder.
     *
     * @param PaymentMethodHolder|array|null $value Данные магазина, для которого сохраняется способ оплаты.
     *
     * @return self
     */
    public function setHolder(mixed $value = null): CreatePaymentMethodRequestBuilder
    {
        $this->currentObject->setHolder($value);

        return $this;
    }

    /**
     * Устанавливает client_ip.
     *
     * @param string|null $value IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
     *
     * @return self
     */
    public function setClientIp(?string $value = null): CreatePaymentMethodRequestBuilder
    {
        $this->currentObject->setClientIp($value);

        return $this;
    }

    /**
     * Устанавливает confirmation.
     *
     * @param AbstractConfirmation|array|null $value Данные, необходимые для инициирования сценария подтверждения привязки.
     *
     * @return self
     */
    public function setConfirmation(mixed $value = null): CreatePaymentMethodRequestBuilder
    {
        $this->currentObject->setConfirmation($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return CreatePaymentMethodRequestInterface|AbstractRequestInterface Инстанс объекта запроса
     *
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);
        }

        return parent::build();
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreatePaymentMethodRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): AbstractRequestInterface
    {
        return new CreatePaymentMethodRequest();
    }
}
