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

namespace YooKassa\Request\PersonalData;

use DateTime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Model\Metadata;
use YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest;

/**
 * Класс, представляющий модель CreatePayoutStatementRecipientPersonalDataRequestBuilder.
 *
 * Класс билдера объектов запросов к API на создание платежа.
 *
 * @example 02-builder.php 210 20 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CreateSbpPayoutRecipientPersonalDataRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var SbpPayoutRecipientPersonalDataRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает фамилию пользователя.
     *
     * @param string $value фамилия пользователя
     */
    public function setLastName(string $value): self
    {
        $this->currentObject->setLastName($value);

        return $this;
    }

    /**
     * Устанавливает имя пользователя.
     *
     * @param string $value имя пользователя
     */
    public function setFirstName(string $value): self
    {
        $this->currentObject->setFirstName($value);

        return $this;
    }

    /**
     * Устанавливает отчество пользователя.
     *
     * @param null|string $value Отчество пользователя
     */
    public function setMiddleName(?string $value): self
    {
        $this->currentObject->setMiddleName($value);

        return $this;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $value Метаданные платежа, устанавливаемые мерчантом
     *
     * @return self Инстанс текущего билдера
     */
    public function setMetadata(mixed $value): self
    {
        $this->currentObject->setMetadata($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return AbstractRequestInterface|SbpPayoutRecipientPersonalDataRequest Инстанс объекта запроса
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return SbpPayoutRecipientPersonalDataRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): SbpPayoutRecipientPersonalDataRequest
    {
        return new SbpPayoutRecipientPersonalDataRequest();
    }
}
