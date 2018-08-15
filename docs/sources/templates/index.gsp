<%include "header.gsp"%>

    <div class="w3-container w3-center" id = "indexCover" style ="padding: 128px 16px">
        <h1 class="w3-margin w3-jumbo">${content.title}</h1>
        <p>${content.subTitle}</p>
        <a href="https://github.com/salesforce/TransmogrifAI" class="w3-button w3-padding-large w3-round-xlarge w3-large w3-border w3-hover-white" id="indexButton">
        Github Link
        </a>
    </div>

	<%include "menu.gsp"%>

    <div class="w3-container" id = "indexContent">

    <p>${content.body}</p>

    </div>

    <!--<p><em>${new java.text.SimpleDateFormat("dd MMMM yyyy", Locale.ENGLISH).format(content.date)}</em></p>-->

    <hr />

<%include "footer.gsp"%>